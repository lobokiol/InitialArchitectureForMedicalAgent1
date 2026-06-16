"""严格子串实体后处理：校验、去重、重叠合并、按位置选主项。"""


def is_valid_span(span: str, query: str) -> bool:
    s = span.strip()
    return bool(s) and s in query


def filter_valid_spans(spans: list[str], query: str) -> list[str]:
    return [s.strip() for s in spans if is_valid_span(s, query)]


def dedupe_by_first_occurrence(spans: list[str], query: str) -> list[str]:
    seen: set[str] = set()
    ordered = sorted(spans, key=lambda s: query.index(s))
    out: list[str] = []
    for span in ordered:
        if span not in seen:
            seen.add(span)
            out.append(span)
    return out


def resolve_overlapping_spans(spans: list[str], query: str) -> list[str]:
    """长 span 吞并被完全包含的短 span，结果按首次出现排序。"""
    valid = [s for s in spans if is_valid_span(s, query)]
    if not valid:
        return []

    valid.sort(key=lambda s: (-len(s), query.index(s)))
    kept: list[str] = []
    for span in valid:
        if any(span in other and span != other for other in kept):
            continue
        kept = [k for k in kept if not (k in span and k != span)]
        kept.append(span)

    kept.sort(key=lambda s: query.index(s))
    return kept


def select_primary_by_position(
    candidates: list[str], query: str
) -> tuple[str | None, list[str]]:
    """
    主项：句中出现最早；同位置取更长 span。
    伴随：其余项按首次出现顺序。
    """
    if not candidates:
        return None, []

    ordered = sorted(candidates, key=lambda c: (query.index(c), -len(c)))
    primary = ordered[0]
    companions = sorted(
        [c for c in candidates if c != primary],
        key=lambda c: query.index(c),
    )
    return primary, companions


def process_spans(spans: list[str], query: str) -> tuple[str | None, list[str]]:
    """校验 → 去重 → 重叠合并 → 选主项。"""
    valid = filter_valid_spans(spans, query)
    deduped = dedupe_by_first_occurrence(valid, query)
    merged = resolve_overlapping_spans(deduped, query)
    return select_primary_by_position(merged, query)
