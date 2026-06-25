# RAG SymptomClarify Recall Gate

**Date:** 2026-06-25  
**Status:** Approved  
**Scope:** Harden `_prefer_symptom_clarify` so unknown symptoms no longer fall through to the first low-confidence `symptomClarify` chunk; route recall misses to a dedicated reject message.

---

## 1. Problem

`rag_symptom_recall_node` uses `_prefer_symptom_clarify`, which:

1. Returns the first `symptomClarify` in hits when **no alias matches** (lines 14–16).
2. Falls back to `hits[0]` when clarify is `None` (line 49).

Observed failure: query「神经疼」with no KB entry recalled **CL0007（脚痛）** at `_score ≈ 1.07`, with runners-up CL0006/CL0008 at ~1.06/1.04 — scores clustered with no alias match.

Adding **CL0015** fixes that specific symptom via alias pass. This change prevents **future** unknown symptoms from the same blind fallback.

When `rag_chunk` is empty, `route_after_rag` currently routes to `dept_disambiguation` (wrong).

---

## 2. Decisions (confirmed)

| Topic | Choice |
|-------|--------|
| Recall miss UX | **A** — dedicated message: 「暂无法识别该症状，请补充具体部位或描述；也可到医院分诊台咨询。」 |
| No-alias fallback | **B** — threshold gate (not alias-required-only) |
| Threshold model | **2 — score + margin** (recommended) |
| RK legacy path | Preserve: if no CL selected and `hits[0].type == "symptom"`, use RK chunk |
| Default thresholds | `RAG_CLARIFY_MIN_SCORE=1.2`, `RAG_CLARIFY_MIN_MARGIN=0.15` (env-overridable) |

---

## 3. `_prefer_symptom_clarify` algorithm

### 3.1 Alias pass (unchanged priority, stricter text)

Match against **`primary_symptom` first**, then full augmented `query`:

```python
def _alias_matches(aliases: list, primary: str, query: str) -> bool:
    for text in (primary, query):
        t = (text or "").strip()
        if not t:
            continue
        for a in aliases:
            if isinstance(a, str) and (a in t or t in a):
                return True
    return False
```

**Pass 1:** iterate `symptomClarify` hits; on alias match → return hit (**no score/margin check**).

### 3.2 Threshold pass (B + score + margin)

If Pass 1 fails:

1. Collect all `symptomClarify` hits from `hits`.
2. Sort by `_score` descending.
3. Let `top` = rank 1, `top_score`, `second_score` (0 if only one CL).
4. `margin = top_score - second_score`
5. Return `top` iff `top_score >= RAG_CLARIFY_MIN_SCORE` **and** `margin >= RAG_CLARIFY_MIN_MARGIN`.
6. Else `None`.

**Remove** the old loop that returns the first `symptomClarify` without checks.

### 3.3 Why margin

Empirical cluster for mis-recall「神经疼」before CL0015:

| id | score |
|----|-------|
| CL0007 | 1.068 |
| CL0006 | 1.061 |
| CL0008 | 1.042 |

`margin ≈ 0.007` — fails `MIN_MARGIN=0.15` even though `top_score > 1.2` would pass a score-only gate.

---

## 4. `rag_symptom_recall_node` selection

```python
clarify = _prefer_symptom_clarify(hits, q, table.primary_symptom or q)
if clarify:
    chunk = clarify
elif hits and hits[0].get("type") == "symptom":
    chunk = hits[0]  # legacy RK path (e.g. foot)
else:
    chunk = None
```

Do **not** use `hits[0]` when it is `symptomClarify` and `_prefer` returned `None`.

Return `{"rag_chunk": None, "rag_chunk_id": None}` when `chunk is None`.

---

## 5. Routing and reject node

### 5.1 `route_after_rag`

```python
def route_after_rag(state: AppState) -> str:
    chunk = state.rag_chunk or {}
    if not chunk:
        return "rag_miss_reject"
    if chunk.get("type") == "symptomClarify":
        return "symptom_clarify"
    return "dept_disambiguation"
```

### 5.2 `rag_miss_reject_node`

New node in `app/graph/nodes/rag_miss_reject.py` (or extend `reject.py` with a flag):

```python
RAG_MISS_MESSAGE = (
    "暂无法识别该症状，请补充具体部位或描述；"
    "也可到医院分诊台咨询。"
)
```

Behavior:

- Append `AIMessage(RAG_MISS_MESSAGE)` to `messages`.
- Call `triage_state_reset_patch()` to clear clarify/dept/rag intermediate state (keep messages).
- Do **not** use `REJECT_MESSAGE`（「请输入症状？」）— user already passed NER/slot gate.

### 5.3 Graph (`builder.py`)

```python
graph.add_node("rag_miss_reject", rag_miss_reject_node)
graph.add_conditional_edges(
    "rag_symptom_recall",
    route_after_rag,
    {
        "symptom_clarify": "symptom_clarify",
        "dept_disambiguation": "dept_disambiguation",
        "rag_miss_reject": "rag_miss_reject",
    },
)
graph.add_edge("rag_miss_reject", END)
```

---

## 6. Configuration

`app/core/config.py`:

```python
RAG_CLARIFY_MIN_SCORE: float = float(os.getenv("RAG_CLARIFY_MIN_SCORE", "1.2"))
RAG_CLARIFY_MIN_MARGIN: float = float(os.getenv("RAG_CLARIFY_MIN_MARGIN", "0.15"))
```

Calibrate via unit tests with mocked hits; optional future script sampling live OpenSearch scores.

---

## 7. Tests

New `tests/test_rag_symptom_recall.py`:

| Case | Input | Expected |
|------|-------|----------|
| `test_alias_hit_ignores_threshold` | CL with alias「神经疼」, low score | returns CL |
| `test_no_alias_clustered_scores` | scores 1.07 / 1.06 | `None` |
| `test_no_alias_high_score_wide_margin` | scores 3.0 / 0.5 | returns top CL |
| `test_route_after_rag_empty_chunk` | `rag_chunk=None` | `rag_miss_reject` |
| `test_route_after_rag_clarify` | `type=symptomClarify` | `symptom_clarify` |
| `test_route_after_rag_rk` | `type=symptom` | `dept_disambiguation` |

Update `tests/test_symptom_clarify_routing.py` if needed for new route key.

---

## 8. Interaction with CL0015 (神经痛)

| Query | Path |
|-------|------|
| 神经疼 (CL0015 in KB) | Pass 1 alias → CL0015 → normal clarify flow |
| 生僻症状, clustered low margin | Pass 2 fails → `rag_miss_reject` |
| 脚疼 (RK or CL0007 alias) | Alias or strong RK top hit → existing flows |

---

## 9. Out of scope

- Changing OpenSearch hybrid pipeline / normalization
- Generic「请描述部位」multi-turn re-prompt (only fixed reject for now)
- Applying gate to `type=symptom` RK chunks

---

## 10. Files to touch

| File | Change |
|------|--------|
| `app/graph/nodes/rag_symptom_recall.py` | Gate logic + chunk selection |
| `app/graph/nodes/rag_miss_reject.py` | New reject node |
| `app/domain/routing.py` | `route_after_rag` empty-chunk branch |
| `app/graph/builder.py` | Wire `rag_miss_reject` |
| `app/core/config.py` | Threshold env vars |
| `tests/test_rag_symptom_recall.py` | Unit tests |
