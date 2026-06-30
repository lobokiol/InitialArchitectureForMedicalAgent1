"""Score batch triage sessions for eval reports.

Chunk / dept relevance scores are derived from production KB jsonl only
(rag_knowledge, disease_kb, rag_department_rules) — no duplicate alias tables.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"
RAG_KB_PATH = ROOT / "sourceData" / "data" / "rag_knowledge.jsonl"
DISEASE_KB_PATH = ROOT / "sourceData" / "data" / "disease_kb.jsonl"
DEPT_RULES_PATH = ROOT / "sourceData" / "data" / "rag_department_rules.jsonl"

INPUT_PATH = EXPORTS / "_eval_input.json"
CHUNK_MAP_PATH = EXPORTS / "_chunk_map.json"
OUT_SCORES_PATH = EXPORTS / "medical_small_100_eval_scores.json"
OUT_SUMMARY_PATH = EXPORTS / "medical_small_100_eval_summary.json"


def short_query(text: str, n: int = 40) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t[:n]


def contains_any(text: str, keywords: list[str] | set[str]) -> bool:
    return any(k in text for k in keywords if k)


def alias_matches(aliases: list, primary: str, query: str) -> bool:
    """Same rule as app.graph.nodes.rag_symptom_recall._alias_matches."""
    for text in (primary, query):
        t = (text or "").strip()
        if not t:
            continue
        for a in aliases:
            if isinstance(a, str) and (a in t or t in a):
                return True
    return False


@lru_cache(maxsize=1)
def load_rag_by_id() -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for line in RAG_KB_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        by_id[d["id"]] = d
    return by_id


@lru_cache(maxsize=1)
def load_dept_keywords() -> dict[str, set[str]]:
    """Dept -> query tokens from disease_kb + dept_rules differential phrases."""
    m: dict[str, set[str]] = {}
    for line in DISEASE_KB_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        names = [d.get("canonical_disease") or ""]
        names.extend(d.get("aliases") or [])
        for dept in d.get("departments") or []:
            bucket = m.setdefault(dept, set())
            for name in names:
                n = (name or "").strip()
                if len(n) >= 2:
                    bucket.add(n)
    for line in DEPT_RULES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        for q in d.get("differential_questions") or []:
            text = (q.get("text") or "").strip()
            if not text:
                continue
            parts = {text}
            for part in re.split(r"[、，,；;或及与和]", text):
                p = part.strip()
                if len(p) >= 2:
                    parts.add(p)
            for dept in (q.get("scores") or {}):
                bucket = m.setdefault(dept, set())
                bucket.update(parts)
    return m


def _chunk_doc(chunk_id: str | None, chunk_map: dict) -> dict:
    if not chunk_id:
        return {}
    rag = load_rag_by_id()
    if chunk_id in rag:
        return rag[chunk_id]
    return chunk_map.get(chunk_id) or {}


def judge_entity_score(case: dict, entity: str | None) -> int:
    query = (case.get("query") or "").strip()
    route = case.get("route")
    symptom = case.get("entity_symptom")
    disease = case.get("entity_disease")

    if not entity:
        return 10 if contains_any(query, ["痛", "痒", "血", "发烧", "头晕", "咳"]) else 0

    score = 70
    if entity in query:
        score += 20
    else:
        score -= 10

    if route == "symptom" and symptom:
        score += 5
    if route == "disease" and disease:
        score += 8

    if symptom and disease:
        if route == "disease" and symptom and symptom == entity:
            score -= 15
        if route == "symptom" and disease and disease == entity:
            score -= 10

    low_quality_entities = {
        "更痛",
        "脸色",
        "幻想",
        "没吃饱",
        "心情不好",
        "精力很旺盛",
        "月经来了五天了",
        "嘴上都是血",
        "大肚子",
    }
    if entity in low_quality_entities:
        score -= 25

    broad_diseases = {"怀孕", "感冒", "心脏病", "内风湿", "支原休", "心肌超微结构"}
    if entity in broad_diseases:
        score -= 15

    return max(0, min(100, int(score)))


def judge_chunk_score(case: dict, chunk_id: str | None, chunk_map: dict) -> int:
    if not chunk_id:
        return 0

    doc = _chunk_doc(chunk_id, chunk_map)
    query = (case.get("query") or "").strip()
    primary = (case.get("entity_symptom") or case.get("entity_disease") or "").strip()

    aliases = doc.get("aliases") or doc.get("alliance") or []
    if alias_matches(aliases, primary, query):
        return 90

    symptom_id = (doc.get("symptom_id") or "").strip()
    if symptom_id and symptom_id in query:
        return 80

    label = (
        doc.get("symptom_id")
        or doc.get("canonical_mechanism")
        or doc.get("keyword")
        or (chunk_map.get(chunk_id) or {}).get("label")
        or ""
    )
    if label and label in query:
        return 80

    if chunk_id.startswith("RK"):
        return 5
    if chunk_id.startswith("CL"):
        return 10
    return 20


def judge_dept_relevance(case: dict) -> int:
    dept = case.get("dept")
    if not dept:
        return 0

    query = (case.get("query") or "").strip()
    keywords = load_dept_keywords().get(dept, set())
    if contains_any(query, keywords):
        return 90

    if dept in ("心内科", "呼吸内科") and contains_any(query, ["胸闷", "气短", "心慌", "心悸"]):
        return 70
    if dept in ("骨科", "风湿免疫科") and contains_any(query, ["疼", "痛"]):
        return 50
    if dept == "神经内科" and contains_any(query, ["头", "睡", "精神"]):
        return 55
    return 25


def judge_chunk_should_recall(case: dict) -> str:
    route = case.get("route")
    query = (case.get("query") or "").strip()
    if route == "disease":
        return "na"
    if route == "symptom":
        return "yes"
    if contains_any(query, ["痛", "痒", "血", "晕", "发烧", "咳", "肿", "不适"]):
        return "yes"
    return "no"


def main() -> None:
    cases = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    chunk_map = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))

    output_rows = []
    entity_scores = []
    chunk_scores = []
    dept_scores_non_null = []

    entity_ok_count = 0
    chunk_present = 0
    chunk_present_good = 0
    should_recall_yes = 0
    recall_good = 0
    dept_non_null = 0
    dept_good = 0

    failure_breakdown = {
        "entity_fail": 0,
        "chunk_missing_when_should_recall": 0,
        "chunk_low_relevance": 0,
        "dept_missing": 0,
        "dept_low_relevance": 0,
    }

    for case in cases:
        idx = case["idx"]
        query = case.get("query") or ""
        entity = case.get("entity_symptom") or case.get("entity_disease")

        entity_score = judge_entity_score(case, entity)
        entity_ok = entity_score >= 70
        if entity_ok:
            entity_ok_count += 1
        else:
            failure_breakdown["entity_fail"] += 1

        chunk_id = case.get("chunk_id")
        chunk_meta = chunk_map.get(chunk_id, {}) if chunk_id else {}
        chunk_symptom = chunk_meta.get("symptom_id")
        case_for_score = {
            "query": query,
            "route": case.get("route"),
            "entity_symptom": case.get("entity_symptom"),
            "entity_disease": case.get("entity_disease"),
        }
        chunk_score = judge_chunk_score(case_for_score, chunk_id, chunk_map)
        should_recall = judge_chunk_should_recall(case)

        if chunk_id:
            chunk_present += 1
            if chunk_score >= 70:
                chunk_present_good += 1

        if should_recall == "yes":
            should_recall_yes += 1
            if chunk_id and chunk_score >= 70:
                recall_good += 1

        if should_recall == "yes":
            if not chunk_id:
                chunk_ok = False
                failure_breakdown["chunk_missing_when_should_recall"] += 1
            else:
                chunk_ok = chunk_score >= 70
                if not chunk_ok:
                    failure_breakdown["chunk_low_relevance"] += 1
        elif should_recall == "no":
            chunk_ok = not bool(chunk_id)
        else:
            chunk_ok = not bool(chunk_id) or chunk_score >= 70

        dept = case.get("dept")
        dept_relevance_score = judge_dept_relevance({"dept": dept, "query": query})
        if dept is None:
            failure_breakdown["dept_missing"] += 1
        else:
            dept_non_null += 1
            dept_scores_non_null.append(dept_relevance_score)
            if dept_relevance_score >= 70:
                dept_good += 1
            else:
                failure_breakdown["dept_low_relevance"] += 1

        output_rows.append(
            {
                "idx": idx,
                "query_short": short_query(query),
                "entity_extracted": entity,
                "entity_score": entity_score,
                "chunk_id": chunk_id,
                "chunk_symptom": chunk_symptom,
                "chunk_score": chunk_score,
                "dept": dept,
                "system_confidence": case.get("dept_confidence"),
                "dept_relevance_score": dept_relevance_score,
                "entity_ok": entity_ok,
                "chunk_ok": chunk_ok,
            }
        )

        entity_scores.append(entity_score)
        chunk_scores.append(chunk_score)

    summary = {
        "entity_accuracy": round(entity_ok_count / len(cases), 4),
        "chunk_precision": round(chunk_present_good / chunk_present, 4) if chunk_present else 0.0,
        "chunk_recall": round(recall_good / should_recall_yes, 4) if should_recall_yes else 0.0,
        "dept_relevance": round(dept_good / dept_non_null, 4) if dept_non_null else 0.0,
        "avg_entity_score": round(sum(entity_scores) / len(entity_scores), 2),
        "avg_chunk_score": round(sum(chunk_scores) / len(chunk_scores), 2),
        "avg_dept_relevance_score_non_null": round(sum(dept_scores_non_null) / len(dept_scores_non_null), 2)
        if dept_scores_non_null
        else 0.0,
        "counts": {
            "total_cases": len(cases),
            "chunk_present_cases": chunk_present,
            "chunk_should_recall_yes_cases": should_recall_yes,
            "dept_non_null_cases": dept_non_null,
        },
        "breakdown_by_failure_type": failure_breakdown,
        "scoring_source": "jsonl:rag_knowledge+disease_kb+rag_department_rules",
    }

    OUT_SCORES_PATH.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {OUT_SCORES_PATH}")
    print(f"Wrote {OUT_SUMMARY_PATH}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
