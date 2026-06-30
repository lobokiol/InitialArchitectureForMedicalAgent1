#!/usr/bin/env python3
"""Build eval table from triage_sessions.db + score + export CSV."""
from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_medical_batch import (  # noqa: E402
    judge_chunk_score,
    judge_dept_relevance,
    judge_entity_score,
)

SRC = ROOT / "sourceData" / "data" / "小医疗数据.json"
DB = ROOT / "data" / "triage_sessions.db"
RAG = ROOT / "sourceData" / "data" / "rag_knowledge.jsonl"
EXPORTS = ROOT / "exports"
USER_ID = "batch-med-small-100"


def load_chunk_map() -> dict:
    m: dict = {}
    for line in RAG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        cid = d["id"]
        label = d.get("symptom_id") or d.get("canonical_mechanism") or d.get("keyword") or d.get("type")
        m[cid] = {
            "type": d.get("type"),
            "symptom_id": d.get("symptom_id"),
            "label": label,
            "aliases": d.get("aliases") or d.get("alliance") or [],
        }
    return m


def parse_confidence(turns: list[dict]) -> tuple[int | str | None, str]:
    """Extract system dept confidence from turns; infer when passed gate but not logged."""
    for t in reversed(turns):
        text = t.get("assistant") or ""
        m = re.search(r"置信度\s*(\d+)\s*分", text)
        if m:
            return int(m.group(1)), "reject文案"
    return None, ""


def infer_confidence(case: dict, turns: list[dict], parsed: int | None, src: str) -> tuple[str, str]:
    if parsed is not None:
        return str(parsed), src or "reject文案"
    route = case.get("route")
    outcome = case.get("outcome")
    dept = case.get("dept")
    if dept and outcome in ("locked", "disease"):
        if route == "disease":
            return "—", "疾病路由(无置信度门禁)"
        return "≥60", "门禁通过(未写入turns)"
    if route == "symptom" and outcome == "unmatched":
        return "—", "未推荐科室"
    if route == "disease" and outcome == "unmatched":
        return "—", "疾病库未命中"
    return "—", "无"


def disease_hit_label(ddr: dict | None) -> tuple[str, str, str]:
    """Return (diseases, kb_depts, hit_status)."""
    if not ddr:
        return "—", "—", "未走疾病链"
    diseases = ddr.get("diseases") or []
    depts = ddr.get("departments") or []
    dis_str = "、".join(diseases) if diseases else "—"
    if not depts:
        return dis_str, "—", "疾病库未命中"
    dept_parts = []
    for item in depts:
        if isinstance(item, dict):
            dept_parts.append(f"{item.get('disease','?')}→{item.get('dept','?')}")
        else:
            dept_parts.append(str(item))
    return dis_str, "；".join(dept_parts), "命中"


def load_cases_from_db() -> list[dict]:
    src_q = [json.loads(l)["questions"] for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM triage_sessions WHERE user_id=? ORDER BY started_at",
        (USER_ID,),
    ).fetchall()
    by_msg: dict[str, dict] = {}
    for r in rows:
        by_msg[r["initial_message"]] = dict(r)

    chunk_map = load_chunk_map()
    cases = []
    for i, q in enumerate(src_q, 1):
        r = by_msg.get(q)
        if not r:
            continue
        snap = json.loads(r.get("state_snapshot_json") or "{}")
        ner = snap.get("ner_result") or {}
        ddr = snap.get("disease_dept_result")
        dcr = snap.get("dept_confidence_result") or {}
        turns = json.loads(r.get("turns_json") or "[]")
        chunk_id = r.get("rag_chunk_id") or snap.get("rag_chunk_id")
        meta = chunk_map.get(chunk_id, {}) if chunk_id else {}
        parsed_conf, conf_src = parse_confidence(turns)
        if dcr.get("score") is not None:
            parsed_conf = int(round(float(dcr["score"])))
            conf_src = "snapshot"
        dis_hit, kb_dept, hit_status = disease_hit_label(ddr)

        case = {
            "idx": i,
            "query": q,
            "route": r.get("actual_route"),
            "outcome": r.get("outcome"),
            "entity_symptom": ner.get("primary_symptom"),
            "entity_disease": ner.get("primary_disease"),
            "chunk_id": chunk_id,
            "chunk_type": meta.get("type"),
            "chunk_label": meta.get("label"),
            "dept": r.get("actual_dept"),
            "dept_confidence_parsed": parsed_conf,
            "disease_hit": dis_hit,
            "disease_kb_dept": kb_dept,
            "disease_hit_status": hit_status,
        }
        conf_display, conf_source = infer_confidence(case, turns, parsed_conf, conf_src)
        case["system_confidence"] = conf_display
        case["confidence_source"] = conf_source

        entity = ner.get("primary_symptom") or ner.get("primary_disease")
        case["entity_extracted"] = entity
        case["entity_score"] = judge_entity_score(
            {
                "query": q,
                "route": case["route"],
                "entity_symptom": ner.get("primary_symptom"),
                "entity_disease": ner.get("primary_disease"),
            },
            entity,
        )
        case["chunk_score"] = judge_chunk_score(
            {
                "query": q,
                "route": case["route"],
                "entity_symptom": ner.get("primary_symptom"),
                "entity_disease": ner.get("primary_disease"),
            },
            chunk_id,
            chunk_map,
        )
        case["dept_relevance_score"] = judge_dept_relevance({"dept": case["dept"], "query": q})
        cases.append(case)
    return cases


def main() -> int:
    cases = load_cases_from_db()
    header = [
        "#", "问题", "路由", "提取实体", "实体分",
        "chunk_id", "chunk类型", "chunk主题", "chunk分",
        "疾病提取", "疾病库科室", "疾病命中",
        "推荐科室", "系统置信度", "置信度说明", "科室关联分", "outcome",
    ]

    def row_for(c: dict) -> list:
        return [
            c["idx"], c["query"], c["route"] or "—",
            c["entity_extracted"] or "—", c["entity_score"],
            c["chunk_id"] or "—", c["chunk_type"] or "—",
            c["chunk_label"] or "—", c["chunk_score"],
            c["disease_hit"], c["disease_kb_dept"], c["disease_hit_status"],
            c["dept"] or "—", c["system_confidence"], c["confidence_source"],
            c["dept_relevance_score"], c["outcome"],
        ]

    EXPORTS.mkdir(parents=True, exist_ok=True)
    (EXPORTS / "_eval_input.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for name in ("medical_small_100_eval_table.csv", "medical_small_100_eval_table_v2.csv"):
        csv_path = EXPORTS / name
        try:
            with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                for c in cases:
                    w.writerow(row_for(c))
            print(f"Wrote {len(cases)} rows -> {csv_path}")
            break
        except PermissionError:
            if name.endswith("_v2.csv"):
                raise
            print(f"Skip locked file {csv_path}, writing v2...")
    rk = [c for c in cases if c.get("chunk_id") == "RK0024"]
    print(f"RK0024 cases: {len(rk)}")
    for c in rk:
        print(f"  #{c['idx']} {c['query'][:40]} chunk主题={c['chunk_label']}")
    conf_filled = sum(1 for c in cases if c["system_confidence"] != "—")
    print(f"置信度有值: {conf_filled}/100 (含≥60推断)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
