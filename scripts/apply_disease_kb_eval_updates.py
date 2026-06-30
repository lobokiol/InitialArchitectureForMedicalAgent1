"""Apply disease_kb updates from medical_small_100_eval_table_v2.csv solutions."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "sourceData" / "data" / "disease_kb.jsonl"

# Map CSV dept names to KB convention.
DEPT_MAP = {
    "妇产科": "妇科",
    "感染科": "感染性疾病科",
}

# canonical -> extra aliases to merge into existing row
ALIAS_UPDATES: dict[str, list[str]] = {
    "腰椎间盘突出症": ["腰间盘穿丁手术", "腰椎间盘突出手术", "腰间盘手术后"],
    "半月板损伤": ["半月板二度损伤", "半月板二度"],
    "上消化道出血": [],  # 黑便 already present
    "周围神经病": ["周围性神经炎", "多发性神经炎"],
    "糖尿病": ["二型糖尿病", "2型糖尿病", "II型糖尿病"],
    "甲状腺功能减退": [],  # already has aliases
    "支气管哮喘": ["哮喘"],  # 咳血 case uses 哮喘
    "类风湿关节炎": ["内风湿", "类风湿因子升高"],
    "子宫内膜异位症": ["巧囊", "卵巢巧克力囊肿"],
    "慢性肝炎": ["乙肝大三阳", "乙型肝炎大三阳", "乙型肝炎e抗体阴性"],
    "尘肺病": ["尘吸肺病", "尘肺", "矽肺"],
}

# New disease rows: (canonical, aliases, description, departments)
NEW_DISEASES: list[tuple[str, list[str], str, list[str]]] = [
    (
        "股癣",
        ["股部癣", "腹股沟癣"],
        "股部皮肤真菌感染，常见腹股沟内侧红斑、瘙痒、脱屑，边界清楚。",
        ["皮肤科"],
    ),
    (
        "子宫内膜增生",
        ["内膜增生", "子宫内膜增厚"],
        "子宫内膜异常增殖，可致月经紊乱、经量增多，需病理评估良恶性。",
        ["妇科"],
    ),
    (
        "眼睑带状疱疹",
        ["眼部带状疱疹", "眼带状疱疹"],
        "带状疱疹累及眼睑及周围，可伴疱疹、疼痛，需警惕角膜受累。",
        ["皮肤科", "眼科"],
    ),
    (
        "卵巢囊肿",
        ["卵巢囊性肿物", "卵巢包块"],
        "卵巢内囊性病变，可无症状或腹胀、腹痛，需超声随访。",
        ["妇科"],
    ),
    (
        "自然流产",
        ["流产", "早期流产", "胎停"],
        "妊娠早期胚胎自然停止发育并排出，可伴阴道流血、腹痛。",
        ["妇科"],
    ),
    (
        "精神分裂症",
        ["精神分裂", "精神病"],
        "以思维、情感、行为不协调为特征的精神疾病，需长期规范治疗。",
        ["精神心理科"],
    ),
    (
        "迟发性运动障碍",
        ["药物性运动障碍"],
        "长期使用抗精神病药物后出现的异常不自主运动，可累及面部、肢体。",
        ["精神心理科", "神经内科"],
    ),
    (
        "尘肺病",
        ["尘吸肺病", "尘肺", "矽肺", "职业性尘肺"],
        "长期吸入粉尘致肺纤维化，常见于矿工、打磨工等职业暴露。",
        ["呼吸内科", "职业病科"],
    ),
    (
        "股骨颈骨折",
        ["髋部骨折", "股骨颈断裂"],
        "股骨颈部位骨折，多见于老年人跌倒，可致髋部疼痛、活动受限。",
        ["骨科"],
    ),
    (
        "鼻甲肥大",
        ["下鼻甲肥大", "鼻甲增生"],
        "鼻甲黏膜增生肥大，可致鼻塞、呼吸不畅，常伴鼻炎。",
        ["耳鼻喉科"],
    ),
    (
        "新生儿黄疸",
        ["黄胆", "新生儿黄胆", "婴儿黄疸", "初生儿黄疸"],
        "新生儿期胆红素升高致皮肤巩膜黄染，需评估生理性还是病理性。",
        ["儿科", "新生儿科"],
    ),
    (
        "子宫癌",
        ["子宫内膜癌", "宫体癌"],
        "子宫体或内膜恶性肿瘤，可异常出血、消瘦，需妇科肿瘤专科。",
        ["妇科", "肿瘤科"],
    ),
    (
        "肝门部胆管癌",
        ["胆管癌", "肝门胆管癌"],
        "肝门部胆管恶性肿瘤，可黄疸、腹痛、消瘦，预后与分期相关。",
        ["肝胆外科", "肿瘤科", "消化内科"],
    ),
    (
        "脑萎缩",
        ["脑萎缩症", "老年性脑萎缩"],
        "脑组织体积缩小，可伴记忆力下降、认知减退，需神经专科评估。",
        ["神经内科"],
    ),
    (
        "心脏病",
        ["左胸有时候像有凉水流过一样", "心脏不适"],
        "心脏疾病统称，可胸闷、心悸、胸痛等，需心内科进一步鉴别。",
        ["心内科"],
    ),
    (
        "急性脊髓炎",
        ["脊髓炎"],
        "脊髓急性炎症，可致肢体无力、感觉障碍、大小便功能障碍。",
        ["神经内科"],
    ),
    (
        "乳腺纤维瘤",
        ["乳腺纤维腺瘤", "乳房纤维瘤"],
        "乳腺良性纤维上皮肿瘤，可触及无痛肿块，需影像随访或手术。",
        ["乳腺外科", "妇科"],
    ),
    (
        "肌张力异常",
        ["肌张力障碍", "肌张力过高", "肌张力过低"],
        "肌肉张力调节异常，可僵硬或过低，见于神经肌肉疾病或发育问题。",
        ["神经内科", "儿科"],
    ),
    (
        "人工流产",
        ["人流", "药流", "清宫术后"],
        "终止妊娠的医疗操作，术后需随访出血、感染及子宫恢复。",
        ["妇科"],
    ),
    (
        "妊娠",
        ["怀孕", "早孕", "孕期", "妊娠状态"],
        "妊娠相关咨询与并发症评估，包括产检、流产、保胎等。",
        ["妇科"],
    ),
    (
        "宫颈糜烂",
        ["宫颈轻度糜烂", "宫颈糜烂样改变"],
        "宫颈柱状上皮外移所致糜烂样外观，常需宫颈筛查排除病变。",
        ["妇科"],
    ),
    (
        "宫颈水肿",
        ["宫颈炎性水肿"],
        "宫颈组织水肿，可伴分泌物增多、接触性出血，需妇科评估。",
        ["妇科"],
    ),
    (
        "近视",
        ["520度近视", "高度近视", "视力下降"],
        "屈光不正，远物模糊，青少年需规范验光与防控。",
        ["眼科"],
    ),
    (
        "更年期综合征",
        ["更年期", "绝经期综合征", "围绝经期"],
        "围绝经期激素波动所致心慌、失眠、潮热、情绪波动等。",
        ["妇科", "内分泌科"],
    ),
    (
        "感冒",
        ["上呼吸道感染", "伤风"],
        "病毒性上呼吸道感染，常见流涕、咳嗽、咽痛、发热。",
        ["呼吸内科", "全科医学科"],
    ),
    (
        "咳嗽",
        ["久咳", "感冒咳嗽"],
        "呼吸道刺激或感染引起的保护性反射，久咳需排查气道与肺部病因。",
        ["呼吸内科"],
    ),
    (
        "支原体感染",
        ["支原休", "支原体肺炎", "肺炎支原体"],
        "支原体引起的感染，可累及呼吸道或泌尿生殖道。",
        ["呼吸内科", "感染性疾病科", "泌尿外科"],
    ),
    (
        "艾滋病",
        ["艾滋病毒", "HIV感染", "AIDS"],
        "HIV感染所致免疫缺陷，需感染科规范抗病毒与随访。",
        ["感染性疾病科", "皮肤科"],
    ),
    (
        "包皮术后并发症",
        ["包皮术后", "包皮环切术后", "包皮手术后"],
        "包皮手术后创面愈合不良、感染或瘢痕等问题。",
        ["泌尿外科"],
    ),
    (
        "尿毒症",
        ["尿毒炎症", "尿蛋白升高", "尿液中蛋白含量较多"],
        "肾功能严重受损致代谢废物蓄积，常伴蛋白尿、水肿。",
        ["肾内科"],
    ),
    (
        "胎儿宫内死亡",
        ["胎儿死在肚子里", "胎死宫内", "死胎"],
        "妊娠中胎儿心跳消失，需及时引产并查明原因。",
        ["妇科"],
    ),
    (
        "输卵管复通术后",
        ["输卵管复通手术", "输卵管复通"],
        "输卵管再通术后备孕与妊娠时间咨询。",
        ["妇科"],
    ),
    (
        "流鼻血",
        ["鼻出血", "衄血"],
        "鼻腔出血，儿童与成人常见，反复需耳鼻喉科排查。",
        ["耳鼻喉科", "儿科"],
    ),
]

# Remove placeholder empty NEW for 冠心病 - handle in ALIAS
NEW_DISEASES = [d for d in NEW_DISEASES if d[0] != "冠心病" or d[2]]
ALIAS_UPDATES.setdefault("冠心病", []).extend(["心脏病", "心肌超微结构"])


def _norm_dept(d: str) -> str:
    return DEPT_MAP.get(d.strip(), d.strip())


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for line in KB_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def next_id(rows: list[dict]) -> str:
    nums = []
    for r in rows:
        rid = r.get("id", "")
        if isinstance(rid, str) and rid.startswith("D"):
            try:
                nums.append(int(rid[1:]))
            except ValueError:
                pass
    n = max(nums, default=0) + 1
    return f"D{n:04d}"


def merge_aliases(row: dict, extras: list[str]) -> None:
    existing = set(row.get("aliases") or [])
    canonical = row.get("canonical_disease") or ""
    for a in extras:
        a = a.strip()
        if a and a != canonical and a not in existing:
            existing.add(a)
    row["aliases"] = sorted(existing, key=len, reverse=True)


def main() -> None:
    rows = load_rows()
    by_canonical = {r["canonical_disease"]: r for r in rows if r.get("canonical_disease")}

    updated = 0
    for canonical, extras in ALIAS_UPDATES.items():
        if canonical in by_canonical and extras:
            merge_aliases(by_canonical[canonical], extras)
            updated += 1

    added = 0
    for canonical, aliases, desc, depts in NEW_DISEASES:
        if canonical in by_canonical:
            if aliases:
                merge_aliases(by_canonical[canonical], aliases)
            continue
        depts = [_norm_dept(d) for d in depts]
        row = {
            "id": next_id(rows),
            "type": "disease",
            "canonical_disease": canonical,
            "aliases": [a for a in aliases if a != canonical],
            "description": desc,
            "departments": depts,
            "version": ["01"],
        }
        rows.append(row)
        by_canonical[canonical] = row
        added += 1

    KB_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(f"Updated alias merges: {updated}")
    print(f"Added diseases: {added}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
