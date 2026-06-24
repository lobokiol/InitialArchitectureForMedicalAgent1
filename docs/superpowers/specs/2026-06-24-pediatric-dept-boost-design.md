# Pediatric Department Boost (age &lt; 14)

**Date:** 2026-06-24  
**Status:** Approved (design §1)  
**Scope:** Boost「儿科」in dept_rules scoring when clarify age indicates a young child.

---

## 1. Problem

Symptom clarify collects age as option buckets. Department locking uses `dept_rules_disambiguation` rule scoring. Some rules include「儿科」in `candidate_departments`, but base scores and differential selections may favor other departments even for infants and young children.

**Goal:** When age clearly indicates a child under 14, give「儿科」a scoring boost if it is already a candidate — without forcing儿科 for all pediatric cases or modifying KB rules.

---

## 2. Decisions (confirmed)

| Topic | Choice |
|-------|--------|
| Priority model | **A** — bonus points + tie-break preference, not forced lock |
| Age buckets | **A (conservative)** — only `0-3个月`, `3个月-1岁`, `2-4岁`, `5-11岁`; **`12-18岁` does not boost** |
| Candidate scope | Only boost if「儿科」already in `active_depts` after sex filter; do not inject儿科 |

---

## 3. Design

### 3.1 Constants (`app/triage/dept_rules_scoring.py`)

```python
PEDIATRIC_DEPT = "儿科"
PEDIATRIC_BOOST = 3.0
PEDIATRIC_AGE_BUCKETS = frozenset({
    "0-3个月", "3个月-1岁", "2-4岁", "5-11岁",
})
```

### 3.2 Helpers

```python
def is_pediatric_age(age_label: str | None) -> bool:
    return bool(age_label and age_label.strip() in PEDIATRIC_AGE_BUCKETS)

def apply_pediatric_boost(
    totals: dict[str, float],
    age_label: str | None,
    active_depts: list[str],
) -> dict[str, float]:
    if not is_pediatric_age(age_label) or PEDIATRIC_DEPT not in active_depts:
        return totals
    out = dict(totals)
    out[PEDIATRIC_DEPT] = out.get(PEDIATRIC_DEPT, 0.0) + PEDIATRIC_BOOST
    return out
```

### 3.3 Integration (`dept_rules_disambiguation_node`)

After `accumulate_scores`, before `lock_department_from_totals`:

```python
age = cs.filled_slots.get("age")
totals = apply_pediatric_boost(totals, age, active_depts)
```

### 3.4 Tie-break (`lock_department_from_totals`)

When multiple departments tie at `best_score` and `is_pediatric_age(age)` is true, prefer `儿科` if it is among `tied`. Pass `age_label` into `lock_department_from_totals` (new optional parameter) or apply preference in the node after lock returns tied fallback.

Implementation note: extend `lock_department_from_totals(..., age_label: str | None = None)` so tie-break logic stays in one place.

### 3.5 Unchanged

- `dept_confidence` LLM gate
- Legacy `dept_disambiguation` path
- `rag_knowledge.jsonl` age options (no split of `12-18岁`)
- Rules JSONL content

---

## 4. Examples

| age | rule candidates | differential | Expected |
|-----|-----------------|--------------|----------|
| `5-11岁` | 消化内科, 普外科, 儿科 | 都没有 | 儿科 +3 on base → likely儿科 |
| `12-18岁` | 消化内科, 儿科 | 都没有 | No boost; existing tie-break |
| `2-4岁` | 眼科, 神经内科 | 视物模糊→眼科 5 | Boost may not beat +5 specialty |
| `5-11岁` | 骨科, 康复医学科 | no儿科 | No boost (儿科 not candidate) |

---

## 5. Testing

**Unit tests** (`tests/test_dept_rules_scoring.py` or extend existing):

- `is_pediatric_age` true/false cases
- `apply_pediatric_boost` adds 3 only when儿科 in active_depts
- `lock_department_from_totals` with tied scores prefers儿科 when pediatric age

---

## 6. Out of scope

- Force `locked_department = 儿科` for all age &lt; 14
- Add儿科 to rules missing it
- NER-parsed age from free text (`5岁` in query) unless already in `filled_slots`
- Changing `PEDIATRIC_BOOST` via env/config (hardcode v1)
