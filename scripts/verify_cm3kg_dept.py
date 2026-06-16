"""Verify CM3KG department routing for a fixed symptom+slots input."""
from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "demo" / "data" / "CM3KG"

INPUT = {
    "chief_symptom": "腹痛",
    "slots": {
        "trigger": "进食后",
        "pain_character": "绞痛",
        "body_part": "上腹",
        "companion": ["腹胀"],
    },
}

TARGET = [
    {"dept": "消化内科", "weight": 0.88},
    {"dept": "普外科", "weight": 0.35},
]


def disease_name(entity: str) -> str:
    return str(entity).replace("[疾病]", "").strip()


def parse_dept_field(val) -> list[str]:
    if pd.isna(val) or not str(val).strip():
        return []
    s = str(val).strip()
    try:
        arr = ast.literal_eval(s)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in re.split(r"[,，;；\s]+", s) if x.strip()]


def leaf_dept(dept_list: list[str]) -> str | None:
    generic = {
        "疾病百科", "内科", "外科", "儿科", "妇产科", "急诊科", "传染科",
        "肿瘤科", "五官科", "其他科室",
    }
    for d in reversed(dept_list):
        if d not in generic:
            return d
    return dept_list[-1] if dept_list else None


def main() -> None:
    kg = pd.read_csv(DATA / "Disease.csv", names=["header", "relation", "tail"], dtype=str)
    med = pd.read_csv(DATA / "medical.csv", dtype=str)

    # symptom -> diseases
    symptom_map = {
        "chief": INPUT["chief_symptom"],
        "companion": INPUT["slots"]["companion"],
        "body_part": INPUT["slots"]["body_part"],
        "extra": ["上腹痛"],  # body_part alias
    }
    weights = {"chief": 3, "companion": 2, "body_part": 2, "extra": 2}

    symptom_to_diseases: dict[str, set[str]] = defaultdict(set)
    for label, syms in [
        ("chief", [symptom_map["chief"]]),
        ("companion", symptom_map["companion"]),
        ("body_part", [symptom_map["body_part"]]),
        ("extra", symptom_map["extra"]),
    ]:
        for sym in syms:
            key = f"{sym}[症状]"
            rows = kg[(kg["header"] == key) & (kg["relation"] == "可能疾病")]
            for tail in rows["tail"].dropna():
                symptom_to_diseases[sym].add(disease_name(tail))

    print("=== Input ===")
    print(INPUT)
    print("\n=== Symptom hit counts ===")
    for sym in ["腹痛", "腹胀", "上腹", "上腹痛"]:
        print(f"  {sym}: {len(symptom_to_diseases.get(sym, set()))} diseases")

    # disease -> dept
    disease_dept: dict[str, set[str]] = defaultdict(set)
    for _, r in kg[kg["relation"].isin(["就诊科室", "三级科室分类"])].iterrows():
        d = disease_name(r["header"])
        tail = str(r["tail"]).strip()
        if r["relation"] == "就诊科室":
            parts = [p.strip() for p in re.split(r"\s{2,}|\s+", tail) if p.strip()]
            dept = leaf_dept(parts) or (parts[-1] if parts else tail)
        else:
            dept = tail
        disease_dept[d].add(dept)

    med_dept: dict[str, str] = {}
    for _, r in med.iterrows():
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        depts = parse_dept_field(r.get("cure_department"))
        if depts:
            med_dept[name] = leaf_dept(depts) or depts[-1]

    disease_score: Counter[str] = Counter()
    for d in symptom_to_diseases.get("腹痛", set()):
        disease_score[d] += weights["chief"]
    for d in symptom_to_diseases.get("腹胀", set()):
        disease_score[d] += weights["companion"]
    for sym in ("上腹", "上腹痛"):
        for d in symptom_to_diseases.get(sym, set()):
            disease_score[d] += weights["body_part" if sym == "上腹" else "extra"]

    dept_score: Counter[str] = Counter()
    missing = 0
    for disease, sc in disease_score.items():
        depts = disease_dept.get(disease)
        if not depts:
            if disease in med_dept:
                depts = {med_dept[disease]}
            else:
                missing += 1
                continue
        for dept in depts:
            dept_score[dept] += sc

    total = sum(dept_score.values()) or 1
    ranked = dept_score.most_common(20)

    print(f"\n=== All departments Top20 (raw vote, {len(disease_score)} diseases) ===")
    for dept, sc in ranked:
        print(f"  {dept}: score={sc}, weight={sc / total:.3f}")

    noise = {
        "急诊科", "儿科", "小儿内科", "传染科", "神经内科", "心内科", "血液科",
        "精神科", "皮肤科", "眼科", "耳鼻喉科", "肿瘤内科", "肿瘤外科", "职业病科",
        "其他综合", "儿科综合", "小儿外科", "新生儿科", "中医科", "康复医学科",
        "营养科", "风湿免疫科", "内分泌科", "肾内科", "泌尿外科", "妇科", "产科",
        "呼吸内科", "胸外科", "心胸外科", "感染科", "儿保门诊",
    }
    filtered = Counter({k: v for k, v in dept_score.items() if k not in noise})
    ftotal = sum(filtered.values()) or 1
    franked = filtered.most_common(10)

    print("\n=== Filtered Top10 (abdominal-relevant) ===")
    for dept, sc in franked:
        print(f"  {dept}: score={sc}, weight={sc / ftotal:.3f}")

    print(f"\nMissing dept mapping: {missing}/{len(disease_score)}")

    print("\n=== Supporting diseases for 消化内科 / 普外科 ===")
    for target in ("消化内科", "普外科"):
        samples = []
        for disease, sc in disease_score.most_common():
            depts = disease_dept.get(disease) or (
                {med_dept[disease]} if disease in med_dept else set()
            )
            if target in depts:
                samples.append((disease, sc))
            if len(samples) >= 10:
                break
        print(f"[{target}] ({sum(1 for d, _ in disease_score.items() if target in (disease_dept.get(d) or ({med_dept[d]} if d in med_dept else set())))} total hits)")
        for d, sc in samples:
            print(f"    {d} (vote={sc})")

    print("\n=== Slot terms in KG ===")
    for term in ("进食后", "绞痛", "上腹", "腹胀"):
        as_symptom = len(kg[kg["header"] == f"{term}[症状]"])
        in_disease_symptom = len(
            kg[(kg["relation"] == "症状") & kg["tail"].str.contains(term, na=False, regex=False)]
        )
        print(f"  {term}: symptom_header={as_symptom}, disease_has_symptom={in_disease_symptom}")

    print("\n=== Target check ===")
    top_raw = [d for d, _ in ranked[:5]]
    top_filt = [d for d, _ in franked[:5]]
    for t in TARGET:
        in_top_raw = t["dept"] in top_raw
        in_top_filt = t["dept"] in top_filt
        w_raw = next((sc / total for d, sc in ranked if d == t["dept"]), 0)
        w_filt = next((sc / ftotal for d, sc in franked if d == t["dept"]), 0)
        print(
            f"  {t['dept']}: raw_top5={in_top_raw} w={w_raw:.3f} | "
            f"filtered_top5={in_top_filt} w={w_filt:.3f} | target={t['weight']}"
        )


if __name__ == "__main__":
    main()
