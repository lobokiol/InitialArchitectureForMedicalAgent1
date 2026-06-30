#!/usr/bin/env python3
"""Build versioned eval table (100 rows incl. skipped) with A/B/D metrics."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_medical_eval_table import (  # noqa: E402
    disease_hit_label,
    infer_confidence,
    load_chunk_map,
    parse_confidence,
)
from scripts.eval_medical_batch import judge_chunk_score, judge_dept_relevance, judge_entity_score
from scripts.eval_round_common import (
    SKIP_INDICES,
    chunk_should_recall,
    compute_metrics,
    dept_relevance_ok,
    is_skipped,
    recommend_ok,
)

SRC = ROOT / "sourceData" / "data" / "小医疗数据.json"
DB = ROOT / "data" / "triage_sessions.db"
EXPORTS = ROOT / "exports"


def load_all_questions() -> list[str]:
    return [
        json.loads(line)["questions"]
        for line in SRC.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def top3_hit(case: dict, chunk_map: dict) -> tuple[str, bool]:
    try:
        from app.infra.opensearch_rag import search_rag_knowledge

        query = (case.get("query") or "").strip()
        entity = (case.get("entity_symptom") or case.get("entity_disease") or "").strip()
        search_q = entity or query
        hits = search_rag_knowledge(search_q, k=8)
        cl_ids = [h.get("id") for h in hits if h.get("type") == "symptomClarify"][:3]
        if not cl_ids:
            return "", False
        for cid in cl_ids:
            if judge_chunk_score(case, cid, chunk_map) >= 70:
                return ",".join(cl_ids), True
        return ",".join(cl_ids), False
    except Exception:
        return "", False


def fail_reason(case: dict) -> str:
    if case.get("skipped"):
        return "路由拒绝-跳过"
    outcome = case.get("outcome") or ""
    route = case.get("route")
    dept = case.get("dept")
    if outcome in ("locked", "disease") and dept:
        if not dept_relevance_ok(case):
            return "科室关联分偏低"
        return ""
    if route == "symptom" and outcome == "unmatched":
        if case.get("chunk_id"):
            if (case.get("chunk_score") or 0) < 70:
                return "chunk误召回"
            if (case.get("dept_confidence_parsed") or 0) < 60:
                return "置信度reject"
            return "未推荐科室"
        return "chunk未召回"
    if route == "disease" and outcome == "unmatched":
        return "疾病库未命中"
    if not route:
        return "路由拒绝"
    if not outcome:
        return "会话未完成"
    return "其他"


def load_db_cases(user_id: str) -> dict[int, dict]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM triage_sessions WHERE user_id=? ORDER BY started_at",
        (user_id,),
    ).fetchall()
    by_idx: dict[int, dict] = {}
    questions = load_all_questions()
    q_to_idx = {q: i + 1 for i, q in enumerate(questions)}
    chunk_map = load_chunk_map()

    for r in rows:
        q = r["initial_message"]
        idx = q_to_idx.get(q)
        if not idx:
            continue
        snap = json.loads(r["state_snapshot_json"] or "{}")
        ner = snap.get("ner_result") or {}
        ddr = snap.get("disease_dept_result")
        dcr = snap.get("dept_confidence_result") or {}
        turns = json.loads(r["turns_json"] or "[]")
        chunk_id = r["rag_chunk_id"] or snap.get("rag_chunk_id")
        meta = chunk_map.get(chunk_id, {}) if chunk_id else {}
        parsed_conf, conf_src = parse_confidence(turns)
        if dcr.get("score") is not None:
            parsed_conf = int(round(float(dcr["score"])))
            conf_src = "snapshot"
        dis_hit, kb_dept, hit_status = disease_hit_label(ddr)

        case = {
            "idx": idx,
            "query": q,
            "route": r["actual_route"],
            "outcome": r["outcome"],
            "entity_symptom": ner.get("primary_symptom"),
            "entity_disease": ner.get("primary_disease"),
            "chunk_id": chunk_id,
            "chunk_type": meta.get("type"),
            "chunk_label": meta.get("label"),
            "dept": r["actual_dept"],
            "dept_confidence_parsed": parsed_conf,
            "disease_hit": dis_hit,
            "disease_kb_dept": kb_dept,
            "disease_hit_status": hit_status,
            "skipped": False,
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
        cmap = chunk_map
        case["chunk_score"] = judge_chunk_score(
            {
                "query": q,
                "route": case["route"],
                "entity_symptom": ner.get("primary_symptom"),
                "entity_disease": ner.get("primary_disease"),
            },
            chunk_id,
            cmap,
        )
        case["dept_relevance_score"] = judge_dept_relevance({"dept": case["dept"], "query": q})
        case_for_score = {
            "query": q,
            "route": case["route"],
            "entity_symptom": ner.get("primary_symptom"),
            "entity_disease": ner.get("primary_disease"),
        }
        top3_ids, top3 = top3_hit(case_for_score, chunk_map)
        case["top3_chunk_ids"] = top3_ids
        case["top3_hit"] = top3
        case["recommend_ok"] = recommend_ok(case)
        case["dept_relevance_ok"] = dept_relevance_ok(case)
        case["round_fail_reason"] = fail_reason(case)
        by_idx[idx] = case
    return by_idx


def synthetic_skipped(idx: int, query: str) -> dict:
    return {
        "idx": idx,
        "query": query,
        "route": "—",
        "outcome": "skipped_pass",
        "entity_extracted": "—",
        "entity_score": 100,
        "chunk_id": "—",
        "chunk_type": "—",
        "chunk_label": "—",
        "chunk_score": 0,
        "disease_hit": "—",
        "disease_kb_dept": "—",
        "disease_hit_status": "跳过",
        "dept": "—",
        "system_confidence": "—",
        "confidence_source": "路由拒绝-自动成功",
        "dept_relevance_score": 100,
        "skipped": True,
        "top3_chunk_ids": "",
        "top3_hit": False,
        "recommend_ok": True,
        "dept_relevance_ok": True,
        "round_fail_reason": "路由拒绝-跳过",
    }


def build_all_cases(user_id: str) -> list[dict]:
    questions = load_all_questions()
    db_cases = load_db_cases(user_id)
    out: list[dict] = []
    for i, q in enumerate(questions, 1):
        if is_skipped(i):
            out.append(synthetic_skipped(i, q))
        elif i in db_cases:
            out.append(db_cases[i])
        else:
            out.append(
                {
                    "idx": i,
                    "query": q,
                    "route": None,
                    "outcome": None,
                    "entity_extracted": None,
                    "entity_score": 0,
                    "chunk_id": None,
                    "chunk_type": None,
                    "chunk_label": None,
                    "chunk_score": 0,
                    "disease_hit": "—",
                    "disease_kb_dept": "—",
                    "disease_hit_status": "未跑",
                    "dept": None,
                    "system_confidence": "—",
                    "confidence_source": "未跑",
                    "dept_relevance_score": 0,
                    "skipped": False,
                    "top3_chunk_ids": "",
                    "top3_hit": False,
                    "recommend_ok": False,
                    "dept_relevance_ok": False,
                    "round_fail_reason": "未跑batch",
                }
            )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--user-id", default=None)
    p.add_argument("--changelog", default="")
    args = p.parse_args()
    ver = args.version
    user_id = args.user_id or f"batch-med-small-100-r{ver}"

    cases = build_all_cases(user_id)
    metrics = compute_metrics(cases)
    metrics["round"] = ver
    metrics["user_id"] = user_id
    metrics["changelog"] = args.changelog
    metrics["skip_indices"] = sorted(SKIP_INDICES)

    header = [
        "#", "skipped", "问题", "路由", "提取实体", "实体分",
        "chunk_id", "chunk类型", "chunk主题", "chunk分",
        "top3_chunk_ids", "top3_hit",
        "疾病提取", "疾病库科室", "疾病命中",
        "推荐科室", "系统置信度", "置信度说明", "科室关联分",
        "recommend_ok", "dept_relevance_ok", "round_fail_reason", "outcome",
    ]

    def row(c: dict) -> list:
        return [
            c["idx"],
            "是" if c.get("skipped") else "否",
            c["query"],
            c.get("route") or "—",
            c.get("entity_extracted") or "—",
            c.get("entity_score", 0),
            c.get("chunk_id") or "—",
            c.get("chunk_type") or "—",
            c.get("chunk_label") or "—",
            c.get("chunk_score", 0),
            c.get("top3_chunk_ids") or "",
            "是" if c.get("top3_hit") else "否",
            c.get("disease_hit", "—"),
            c.get("disease_kb_dept", "—"),
            c.get("disease_hit_status", "—"),
            c.get("dept") or "—",
            c.get("system_confidence", "—"),
            c.get("confidence_source", "—"),
            c.get("dept_relevance_score", 0),
            "是" if c.get("recommend_ok") else "否",
            "是" if c.get("dept_relevance_ok") else "否",
            c.get("round_fail_reason") or "",
            c.get("outcome") or "—",
        ]

    EXPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = EXPORTS / f"medical_small_100_eval_table_v{ver}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for c in cases:
            w.writerow(row(c))

    summary_path = EXPORTS / f"medical_small_100_eval_summary_v{ver}.json"
    summary_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXPORTS / "_eval_input.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.changelog:
        cl_path = EXPORTS / f"round_changelog_v{ver}.md"
        cl_path.write_text(args.changelog, encoding="utf-8")

    print(f"Wrote {csv_path} ({len(cases)} rows)")
    print(f"Wrote {summary_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["target_A_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
