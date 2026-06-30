# RAG 妇科/妊娠 + 儿科导诊知识库设计

**日期**: 2026-06-30  
**状态**: 已批准，待实现  
**范围**: 仅修改 `rag_knowledge.jsonl`、`rag_department_rules.jsonl`；重索引后验证 12 条 symptom+unmatched 案例

---

## 1. 目标

修复 batch-100 评测中 **route=symptom + outcome=unmatched** 的 12 条案例（妇科/妊娠 8 条 + 儿童 4 条），使其经症状链锁定 **妇科** 或 **儿科**。

不在本轮范围：其余 9 条 symptom+unmatched、disease 路由未命中等。

---

## 2. 方案

复刻 `CL0010`（眼睛不适）模式：

- 新增 `symptomClarify`：`required_slots` 仅 `age` + `sex`，`default_location` 跳过部位反问
- 新增 `rag_department_rules`：`symptom_id` + `location` 精确匹配，`candidate_departments` 单科室

| ID | symptom_id | default_location | 目标科室 |
|----|------------|------------------|----------|
| CL0016 | 妇科不适 | 妇科 | 妇科 |
| CL0017 | 儿童症状 | 儿童 | 儿科 |
| RK0170 | 妇科不适 | 妇科 | 妇科 |
| RK0171 | 儿童症状 | 儿童 | 儿科 |

---

## 3. 目标案例

**妇科/妊娠**: #6, #7, #8, #22, #26, #81, #84, #97  
**儿童**: #10, #48, #87, #98

---

## 4. 数据变更

### 4.1 CL0016 别名（节选）

月经、怀孕、胎儿、胎心、流产、清宫、外阴瘙痒、白带、子宫、卵巢、宫颈、诊刮、子宫内膜、盆腔、经量、停经、备孕

### 4.2 CL0017 别名（节选）

宝宝、婴儿、幼儿、小孩、孩子、儿童、小儿、打嗝、磨牙、流鼻血、夜醒、睡觉不踏实、免疫力、没吃饱

### 4.3 CL0013 调整

从 `CL0013` 移除 `外阴痒`，避免与 `CL0016` 别名竞争。

### 4.4 科室规则

`RK0170` / `RK0171` 的 `candidate_departments` 各仅含一个科室；`differential_questions` 全部指向该科室（score 5）。

---

## 5. 验证

1. `python sourceData/opensearch_rag_kb.py`
2. `python sourceData/opensearch_dept_rules.py`
3. 对 12 条用例跑 `/chat`（妇科案例 sex=女，儿童案例 age 选儿科段）
4. 验收：`outcome=locked`，`actual_dept` 为 妇科 或 儿科

---

## 6. 风险

- 别名过宽可能误抢其他 CL；通过从 CL0013 剥离外阴别名缓解
- batch 脚本 `pick_reply` 默认 sex=男，妇科全测需用女性 persona 或手工指定
