# 脚部导诊 Golden Set 评测设计

## 目标

为「脚部 → 症状/疾病 → 科室」导诊链路建立可回归的 Golden Set 评测体系：固定跑分脚本、改库必重跑、记录版本哈希，支撑脚部试点发布门禁。

## 决策摘要

| 项 | 选择 |
|----|------|
| 跑分层级 | offline → graph → 可选 `--live` |
| 数据组织 | 单文件 `demo/data/foot_triage_golden.jsonl` + `subset` 字段 |
| B / F 关系 | 不重叠：B 70 条全单轮；F 10–15 条纯多轮 |
| D 负例 | 统一 `triage_route == "reject"` + 固定文案「请输入症状？」 |
| 科室判定 | 严格单值 `expect_dept` |
| 版本追溯 | 内容 SHA256 前 8 位 + JSON 报告 |
| 实现方案 | 统一 Runner `scripts/run_foot_triage_eval.py`（方案 1） |

## Golden Set 结构

### 文件

- **主文件**：`demo/data/foot_triage_golden.jsonl`
- **废弃**：`demo/data/foot_symptom_eval.jsonl`（迁移后删除或保留 thin wrapper 指向 golden）

### 单轮 schema

```json
{
  "id": "FTB001",
  "subset": "B",
  "message": "脚出汗",
  "expect_route": "symptom",
  "expect_chunk_id": "RK0010",
  "expect_dept": "皮肤科",
  "expect_emergency": false,
  "tags": ["alias", "semantic"]
}
```

字段说明：

- `subset`：`A` | `B` | `C` | `D` | `E` | `F`（每条仅一个）
- `expect_route`：`symptom` | `disease` | `reject`
- `expect_chunk_id`：A 子集必填；B 可选（标注用）
- `expect_dept`：B/C/E/F 必填；D 不填
- `expect_emergency`：E 子集为 `true`

### 多轮 schema（subset F）

```json
{
  "id": "FTF001",
  "subset": "F",
  "expect_route": "symptom",
  "expect_dept": "血管外科",
  "expect_emergency": false,
  "turns": [
    {"role": "user", "message": "脚肿"},
    {"role": "assistant", "expect_asking": true},
    {"role": "user", "message": "两脚都肿一按一个坑"}
  ],
  "tags": ["multiturn", "dept_disambiguation"]
}
```

### 子集规模（合计约 105–125 unique 条）

| subset | 条数 | 测什么 | 现状 |
|--------|------|--------|------|
| A 症状召回 | 50 | offline top1 `chunk_id` | 已有 50，迁移 |
| B 科室正确 | 50 + 20 难例 | graph `locked_department` | live 部分有，无统一指标 |
| C 疾病直达 | 15–20 | 病名/别名 → 首科室 | 仅 6 条（`FOOT_DISEASE_ACCEPTANCE`） |
| D 负例/拒答 | 20–30 | reject + 无误推 | 无 |
| E 急诊 | 5–10 | 必须出「急诊」 | 1 条，live 曾 skip |
| F 多轮 | 10–15 | 澄清后科室 | 无 |

### A/B 同 message 策略

- A：FT001–FT050，`subset: "A"`，含 `expect_chunk_id` + `expect_dept`（dept 不参与 A 指标）
- B 基础 50：FTB001–FTB050，与 A 同 message、同 expect，仅 id/subset 不同
- B 难例 20：FTB051–FTB070，输入模糊但期望**单轮** lock（测 scoring/消歧阈值）
- F 与 B 难例**不重叠**

## Runner 架构

### 入口

```bash
# 改库后一键回归
uv run python scripts/run_foot_triage_eval.py --reindex --all

# 分层
uv run python scripts/run_foot_triage_eval.py --offline          # A
uv run python scripts/run_foot_triage_eval.py --graph            # B/C/D/E/F
uv run python scripts/run_foot_triage_eval.py --graph --live     # + /chat 冒烟

# 按子集
uv run python scripts/run_foot_triage_eval.py --subset B,E
```

### 三层断言

| 层级 | 子集 | 断言来源 | 依赖 |
|------|------|----------|------|
| `--offline` | A | OpenSearch recall + alliance rerank（同 `rag_symptom_recall`） | OpenSearch + embedding |
| `--graph` | B/C/D/E/F | LangGraph invoke 后读 AppState | OpenSearch + NER + LLM |
| `--live` | B/C/E（可选全量） | `/chat` reply | FastAPI 已启动 |

`--live` 不参与门槛判定，仅作端到端冒烟报告。

### 各子集 graph 断言

| subset | pass 条件 |
|--------|-----------|
| **A**（offline） | top1 chunk id == `expect_chunk_id` |
| **B** | `locked_department == expect_dept` |
| **C** | `disease_dept_result.departments[0].dept == expect_dept`（疾病链不设 `locked_department`） |
| **D** | `intent_result.triage_route == "reject"`；末条 reply == `请输入症状？`；`locked_department == null` |
| **E** | `locked_department == "急诊"`（100% 红线，禁止 skip） |
| **F** | 按 `turns[]` 多轮 invoke 后 `locked_department == expect_dept` |

### F 多轮执行

1. 新 thread（MemorySaver），发送 `turns[0].message` → invoke
2. 若 `dept_state.status == "asking"` → 追加下一 user turn → invoke
3. 最多 4 轮；超时未 lock 判 fail
4. 可选断言 assistant 步 `expect_asking: true`

复用现有 `route_after_trim`：用户回复科室反问时跳过 decision，继续 `dept_disambiguation`。

### 模块结构

```
scripts/run_foot_triage_eval.py
scripts/foot_eval/
  loader.py
  offline.py
  graph_runner.py
  live.py
  metrics.py
  version.py
  report.py
```

## 指标门槛

| 指标 | 子集 | 门槛 | 计算 |
|------|------|------|------|
| 症状召回准确率 | A | ≥ 90% | offline top1 命中 |
| 科室准确率 | B + F | ≥ 85% | graph `locked_department` 命中 |
| 疾病直达准确率 | C | ≥ 90% | graph 首科室命中 |
| 急诊召回 | E | **100%** | `locked_department == "急诊"` |
| 误推率 | D | ≤ 5% | D 中 `locked_department != null` 的比例 |
| 拒答准确率 | D | ≥ 90% | route==reject 且 reply 正确 |

任一红线未达标 → `sys.exit(1)`。

## 版本报告与发布检查

### 报告输出

```
reports/foot_eval_<YYYYMMDD_HHMMSS>.json
reports/foot_eval_latest.json
```

报告含：三份 jsonl 的 SHA256 前 8 位、OpenSearch 条数、各 subset 指标、failures 列表、`overall_ok`。

### `--reindex` 发布检查单

| 检查项 | 规则 | 失败 |
|--------|------|------|
| OpenSearch 可达 | `wait_for_opensearch` | exit 1 |
| rag 条数 | index count == jsonl 行数 | exit 1 |
| disease 条数 | index count == jsonl 行数 | exit 1 |
| embedding 齐全 | rag 每条 embedding 非空 | exit 1 |
| golden chunk 引用 | `expect_chunk_id` ∈ rag ids | warn |
| C 疾病引用 | expect 对应 disease 在 kb 存在 | warn |

**同步策略**：jsonl 为 source of truth；改库后必须 `--reindex` 再跑分。

## D 子集标注范围

统一拒答，用例须使 NER + 路由产出 `reject`：

- 空输入、问候、预约/流程、无脚无关闲聊
- 示例：`""`、`"你好"`、`"怎么挂号"`、`"今天天气"`、`"帮我查报告"`
- 不含「纯流程但 NER 非 reject」的边界（除非先改路由）

## 迁移计划

1. 新建 `foot_triage_golden.jsonl`
2. 迁移 FT001–FT050 → subset A
3. 复制为 FTB001–FTB050 → subset B
4. 新增 B 难例 20、C 15–18、D 25、E 7、F 12（实施阶段逐条标注）
5. 吸收 `demo/opensearch_disease_kb.py` 中 `FOOT_DISEASE_ACCEPTANCE` 6 条进 C
6. 废弃 `foot_symptom_eval.jsonl`
7. `test_foot_symptom_eval.py` 保留 thin wrapper 调新 runner（兼容 1 个版本）

## 工作流

```bash
# 改 demo/data/*.jsonl 后
uv run python scripts/run_foot_triage_eval.py --reindex --all

# 快速迭代
uv run python scripts/run_foot_triage_eval.py --graph --subset B

# 发版前
uv run python scripts/run_foot_triage_eval.py --reindex --all --live
```

## 不在范围

- 非脚部 body_part 扩展（后续独立 golden）
- pytest 全量替代（runner 为主，pytest 可后补薄封装）
- 修改 NER/路由行为（评测先固定现状；D 标注与路由不一致时单独开 issue）
