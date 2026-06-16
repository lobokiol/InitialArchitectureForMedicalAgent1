from app.ner.models import NERExtractOutput

# 别名 → 标准主症
CHIEF_SYMPTOM_ALIASES: dict[str, str] = {
    "肚脐上面": "肚脐上方疼痛",
    "肚脐上方": "肚脐上方疼痛",
    "上腹部": "肚脐上方疼痛",
    "绞着疼": "绞痛",
    "绞疼": "绞痛",
    "一阵一阵绞着疼": "绞痛",
    "饭后腹胀": "饭后腹胀",
    "进食后": "饭后",
    "吃饭后": "饭后",
    "吃得越多越瘦": "越吃越瘦",
    "脾气变得很差": "脾气差",
    "脾气也变得很差": "脾气差",
    **{
        a: "心悸"
        for a in (
            "心慌", "心里发慌", "心跳厉害", "心里难受", "心累", "落空感", "心跳快",
            "心跳加速", "心跳不齐", "心律不齐", "心扑通扑通跳", "心咚咚跳", "心突突跳",
            "心乱", "心焦", "心跳声大", "能听见心跳", "早搏感", "漏跳感",
        )
    },
}

def _norm_chief(value: str) -> str:
    v = value.strip()
    return CHIEF_SYMPTOM_ALIASES.get(v, v)


def _norm_disease(value: str) -> str:
    return value.strip().replace("有", "").replace("是不是", "").strip()


def dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def normalize_output(data: NERExtractOutput) -> NERExtractOutput:
    chief = dedupe_preserve(_norm_chief(x) for x in data.chief_symptoms)
    diseases = dedupe_preserve(_norm_disease(x) for x in data.diseases)
    return NERExtractOutput(chief_symptoms=chief, diseases=diseases)


def merge_output(base: NERExtractOutput, extra: NERExtractOutput) -> NERExtractOutput:
    return normalize_output(
        NERExtractOutput(
            chief_symptoms=base.chief_symptoms + extra.chief_symptoms,
            diseases=base.diseases + extra.diseases,
        )
    )


def output_from_hints(hints: dict[str, list[str]], query: str) -> NERExtractOutput:
    """词典 hints 兜底分类。"""
    chief: list[str] = list(hints.get("主症", []))
    diseases: list[str] = list(hints.get("疾病", []))

    q = query
    if any(k in q for k in ("疼", "痛", "绞")) and "肚脐" in q:
        if "肚脐上方疼痛" not in chief:
            chief.append("肚脐上方疼痛")
    if "绞着疼" in q or "绞痛" in q:
        if "绞痛" not in chief:
            chief.append("绞痛")
    if "饭后" in q and "腹胀" in q:
        if "饭后腹胀" not in chief:
            chief.append("饭后腹胀")
    elif "腹胀" in q and "腹胀" not in chief:
        chief.append("腹胀")
    if "心慌" in q and "心慌" not in chief:
        chief.append("心慌")
    if "手抖" in q and "手抖" not in chief:
        chief.append("手抖")
    if "越吃越瘦" in q or "吃得越多越瘦" in q:
        if "越吃越瘦" not in chief:
            chief.append("越吃越瘦")
    if "脾气" in q and "脾气差" not in chief:
        chief.append("脾气差")
    if "胃炎" in q and "胃炎" not in diseases:
        diseases.append("胃炎")

    return normalize_output(NERExtractOutput(chief_symptoms=chief, diseases=diseases))
