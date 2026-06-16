"""Load disease_kb.jsonl — single source for disease names, aliases, departments."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISEASE_KB_PATH = _REPO_ROOT / "demo" / "data" / "disease_kb.jsonl"
DEFAULT_TRIAGE_TEMPLATES_PATH = _REPO_ROOT / "demo" / "data" / "triage_templates.jsonl"


def _disease_kb_path() -> Path:
    import os

    custom = os.getenv("DISEASE_KB_PATH")
    return Path(custom) if custom else DEFAULT_DISEASE_KB_PATH


@lru_cache(maxsize=1)
def load_disease_kb_rows() -> list[dict[str, Any]]:
    path = _disease_kb_path()
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def enrich_disease_doc(doc: dict[str, Any]) -> dict[str, Any]:
    parts = [
        doc.get("canonical_disease", ""),
        doc.get("description", ""),
        " ".join(doc.get("aliases") or []),
    ]
    out = dict(doc)
    out["search_text"] = " ".join(p for p in parts if p)
    return out


@lru_cache(maxsize=1)
def disease_term_index() -> dict[str, dict[str, Any]]:
    """Map alias or canonical term → disease row."""
    index: dict[str, dict[str, Any]] = {}
    for row in load_disease_kb_rows():
        canonical = row.get("canonical_disease") or ""
        if canonical:
            index[canonical] = row
        for alias in row.get("aliases") or []:
            if alias:
                index[str(alias)] = row
    return index


def resolve_disease_term(term: str) -> dict[str, Any] | None:
    return disease_term_index().get((term or "").strip())


def lookup_departments_local(disease_terms: list[str]) -> list[dict[str, str]]:
    """疾病/别名 → 科室，去重保序；输出 canonical 病名。"""
    seen_depts: set[str] = set()
    results: list[dict[str, str]] = []
    for term in disease_terms:
        row = resolve_disease_term(term)
        if not row:
            continue
        canonical = row.get("canonical_disease") or term
        for dept in row.get("departments") or []:
            if dept and dept not in seen_depts:
                seen_depts.add(dept)
                results.append({"disease": canonical, "dept": dept})
    return results


@lru_cache(maxsize=1)
def disease_catalog_terms() -> list[str]:
    """All canonical + aliases for NER substring scan (longest first)."""
    terms: set[str] = set()
    for row in load_disease_kb_rows():
        canonical = row.get("canonical_disease")
        if canonical:
            terms.add(str(canonical))
        for alias in row.get("aliases") or []:
            if alias:
                terms.add(str(alias))
    return sorted(terms, key=len, reverse=True)


@lru_cache(maxsize=1)
def symptom_catalog_terms() -> list[str]:
    """Symptom terms from triage_templates.jsonl for NER fallback."""
    path = DEFAULT_TRIAGE_TEMPLATES_PATH
    if not path.exists():
        return []
    terms: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        canonical = row.get("canonical_symptom")
        if canonical:
            terms.add(str(canonical))
        for alias in row.get("alliance") or []:
            if alias:
                terms.add(str(alias))
    return sorted(terms, key=len, reverse=True)
