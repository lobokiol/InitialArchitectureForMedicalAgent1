import difflib
import re


def normalize_string(text: str) -> str:
    """中文友好清洗：小写、保留 CJK/字母/数字、规范空格。"""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\u4e00-\u9fff]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_cjk_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def fuzzy_find_optimized(query: str, keys: list[str], threshold: float = 0.85) -> list[str]:
    """
    从 query 中召回 entity_list 候选（Hints）。
    在原有英文逻辑上增加中文子串匹配。
    """
    matched_keys: set[str] = set()

    norm_query = normalize_string(query)
    if not norm_query:
        return []

    query_no_space = norm_query.replace(" ", "")
    query_len = len(norm_query)
    unique_keys = set(keys)

    for key in unique_keys:
        norm_key = normalize_string(key)
        if not norm_key:
            continue

        key_len = len(norm_key)
        key_is_cjk = any(_is_cjk_char(c) for c in norm_key)

        # 中文短词：子串匹配（去空格）
        if key_is_cjk and key_len <= 4:
            key_compact = norm_key.replace(" ", "")
            query_compact = query_no_space
            if key_compact and key_compact in query_compact:
                matched_keys.add(key)
            continue

        # 英文短词：词边界或数字词
        if not key_is_cjk and key_len < 4:
            has_digit = any(char.isdigit() for char in norm_key)
            if has_digit:
                if norm_key in norm_query:
                    matched_keys.add(key)
            else:
                pattern = r"\b" + re.escape(norm_key) + r"\b"
                if re.search(pattern, norm_query):
                    matched_keys.add(key)
            continue

        # 长词快路径：精确包含
        if norm_key in norm_query:
            matched_keys.add(key)
            continue

        # 去空格包含（BMC GP / BMCGP）
        key_no_space = norm_key.replace(" ", "")
        if key_len > 3 and key_no_space in query_no_space:
            matched_keys.add(key)
            continue

        # 中文长词：紧凑子串
        if key_is_cjk and key_no_space in query_no_space:
            matched_keys.add(key)
            continue

        # 模糊滑动窗口（偏英文）
        if key_is_cjk or key_len > query_len:
            continue

        key_chars = set(norm_key)
        query_chars = set(norm_query)
        if len(key_chars.intersection(query_chars)) / len(key_chars) < 0.6:
            continue

        for i in range(0, query_len - key_len + 1):
            window = norm_query[i : i + key_len]
            if window[0] != norm_key[0]:
                continue
            ratio = difflib.SequenceMatcher(None, norm_key, window).ratio()
            if ratio >= threshold:
                matched_keys.add(key)
                break

    # 长词吃短词
    sorted_matched = sorted(matched_keys, key=len, reverse=True)
    filtered_keys: list[str] = []
    for current_key in sorted_matched:
        current_norm = normalize_string(current_key).replace(" ", "")
        is_subset = False
        for kept_key in filtered_keys:
            kept_norm = normalize_string(kept_key).replace(" ", "")
            if current_norm and current_norm in kept_norm:
                is_subset = True
                break
        if not is_subset:
            filtered_keys.append(current_key)

    return filtered_keys
