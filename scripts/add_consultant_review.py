#!/usr/bin/env python3
"""Add GPT-5.5 consultant review columns to the medical eval table."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "exports" / "medical_small_100_eval_table_v2.csv"
REVIEW_COLUMNS = ["顾问评分", "顾问通过", "不通过理由"]

# Scores answer: "If the patient asked this question, should this be the sole
# recommended first-visit department?" Passing requires score > 90.
CONSULTANT_SCORES = {
    1: 95,
    2: 97,
    3: 92,
    4: 96,
    5: 98,
    6: 96,
    7: 96,
    8: 98,
    9: 97,
    10: 95,
    11: 82,
    12: 98,
    13: 80,
    14: 98,
    15: 98,
    16: 75,
    17: 98,
    18: 60,
    19: 96,
    20: 94,
    21: 86,
    22: 96,
    23: 98,
    24: 70,
    25: 96,
    26: 98,
    27: 80,
    28: 98,
    29: 88,
    30: 78,
    31: 35,
    32: 55,
    33: 85,
    34: 98,
    35: 99,
    36: 91,
    37: 70,
    38: 98,
    39: 92,
    40: 98,
    41: 55,
    42: 98,
    43: 97,
    44: 98,
    45: 92,
    46: 78,
    47: 95,
    48: 92,
    49: 92,
    50: 93,
    51: 68,
    52: 95,
    53: 78,
    54: 99,
    55: 78,
    56: 97,
    57: 82,
    58: 98,
    59: 98,
    60: 98,
    61: 40,
    62: 98,
    63: 92,
    64: 25,
    65: 72,
    66: 98,
    67: 96,
    68: 97,
    69: 95,
    70: 96,
    71: 98,
    72: 80,
    73: 95,
    74: 45,
    75: 99,
    76: 98,
    77: 84,
    78: 82,
    79: 97,
    80: 96,
    81: 98,
    82: 99,
    83: 62,
    84: 98,
    85: 98,
    86: 95,
    87: 95,
    88: 92,
    89: 93,
    90: 98,
    91: 98,
    92: 91,
    93: 91,
    94: 98,
    95: 98,
    96: 95,
    97: 93,
    98: 96,
    99: 99,
    100: 96,
}

FAILURE_REASONS = {
    11: "眼睑带状疱疹累及眼睑，首诊更应优先眼科评估眼部受累风险，皮肤科不是唯一最优推荐。",
    13: "单纯体型或腹部肥胖诉求缺少明确疾病线索，全科可初筛但不是高置信首诊科室。",
    16: "脚汗、异味更偏皮肤科或足部真菌/多汗评估，全科医学科过于泛化。",
    18: "肾阴虚和麒麟丸属于中医辨证及中成药咨询，泌尿外科不是最合适首诊。",
    21: "76岁老人服用冬虫夏草后头晕，需先考虑药物反应、血压或急症风险，神经内科不是唯一首诊。",
    24: "16岁性欲和性健康咨询更适合妇产科、青春期门诊或心理/性健康咨询，全科不是高置信唯一科室。",
    27: "痰中带血可能来自呼吸道或鼻咽部，耳鼻喉科可选但不能作为唯一高置信首诊。",
    29: "夜汗和四肢无力可能涉及感染、内分泌、结核等多系统问题，全科可初筛但置信度不足。",
    30: "体检胸片右心膈小结节影更应结合影像科/呼吸内科复核，全科医学科不是最直接科室。",
    31: "问题语境像妊娠相关腹部包块和胎儿发育担忧，并非婴幼儿就诊，儿科明显不匹配。",
    32: "主诉以长期胃痛为核心，首诊应偏消化内科，风湿免疫科不是合理首选。",
    33: "嗜睡原因复杂，全科可初筛但需要睡眠、神经、内分泌等鉴别，作为唯一推荐置信度不足。",
    37: "儿童磨牙通常先看口腔科或儿科，耳鼻喉科不是最匹配的唯一首诊。",
    41: "描述缺少疼痛部位，仅因看书电脑加重不能直接归神经内科，推荐依据不足。",
    46: "幻想、健忘、头疼包含精神心理症状，精神心理科通常比神经内科更合适首诊。",
    51: "肺癌术后刀口和肋骨异常更应考虑胸外科或肿瘤术后随访，呼吸内科不是最优。",
    53: "婴幼儿烫伤水泡应优先烧伤科/急诊外科处理，单纯皮肤科可能延误创面评估。",
    55: "肝门部胆管癌预后咨询更偏肝胆外科或肿瘤科，消化内科不是唯一最合适科室。",
    57: "药品剂量换算问题不构成明确科室导诊，全科医学科推荐置信度不足。",
    61: "14岁高度近视保护眼睛应首诊眼科，全科医学科明显不匹配。",
    64: "牙疼和牙龈肿痛应首诊口腔科，神经内科明显不匹配。",
    65: "腕背部筋疼更偏骨科、手外科或运动医学，风湿免疫科不是首选。",
    72: "艾滋病毒传播咨询更适合感染科或性病/皮肤性病科，全科不是高置信唯一推荐。",
    74: "类风湿因子升高应优先风湿免疫科，全科医学科不是最合适首诊。",
    77: "喝水口腔疼痛更偏口腔科或耳鼻喉科，全科医学科不是最直接推荐。",
    78: "支原体感染需结合部位，常见为呼吸、泌尿生殖或感染科方向，全科推荐过泛。",
    83: "孕前三个月同房咨询属于孕期保健问题，妇产科比全科医学科更匹配。",
}


def main() -> int:
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = [name for name in reader.fieldnames or [] if name not in REVIEW_COLUMNS]

    missing = sorted(set(range(1, 101)) - set(CONSULTANT_SCORES))
    if missing:
        raise SystemExit(f"Missing consultant scores for rows: {missing}")

    for row in rows:
        idx = int(row["#"])
        score = CONSULTANT_SCORES[idx]
        row["顾问评分"] = str(score)
        row["顾问通过"] = "通过" if score > 90 else "不通过"
        row["不通过理由"] = "" if score > 90 else FAILURE_REASONS[idx]

    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for score in CONSULTANT_SCORES.values() if score > 90)
    print(f"Updated {CSV_PATH}")
    print(f"Rows: {len(rows)}, passed: {passed}, failed: {len(rows) - passed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
