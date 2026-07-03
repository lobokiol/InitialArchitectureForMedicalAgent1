# CI 单元测试合并门禁设计

**日期**: 2026-07-04  
**状态**: 待实现  
**范围**: 在 GitHub Actions 中运行 `tests/test_*.py` 单元测试，并通过 Branch Protection 将其设为合入 `main` 的必过检查。

---

## 1. 背景与目标

### 1.1 现状

- 远程仓库：`github.com/lobokiol/InitialArchitectureForMedicalAgent1`（GitHub）
- **尚无 CI 配置**（无 `.github/workflows/`）
- `tests/` 目录共 **124** 个 pytest 用例（`test_*.py`），本地全绿，约 **23 秒**
- 多数测试通过 monkeypatch/mock 隔离 LLM、MCP、Redis、OpenSearch，**无需外部服务**
- `pyproject.toml` 已配置 pytest：`testpaths = ["tests"]`、`python_files = "test_*.py"`、`asyncio_mode = auto`
- `app/core/config.py` 在 import 时 **强制要求** `DASHSCOPE_API_KEY`（无 TTY 时抛 `RuntimeError`），CI 须注入 dummy 值

### 1.2 目标

| 能力 | 说明 |
|------|------|
| PR 自动跑测 | 指向 `main` 的 PR 触发 workflow |
| 合并门禁 | Branch Protection 要求 `unit-tests` status check 通过 |
| 范围明确 | 仅 `tests/test_*.py` 单元测试；不含评估脚本 |
| 零 Secret | 测试全 mock，不依赖真实 API Key 或 `.env` 文件 |
| 本地一致 | 本地 `pytest tests/ -q` 与 CI 命令一致 |

### 1.3 明确不做

- ruff / mypy lint 门禁
- `tests/run_eval.py`、`tests/test_langsmith.py`（非 pytest 用例）
- `batch_100_cases.json` 批量评估
- OpenSearch / Redis Docker 服务
- 多 Python 版本矩阵（固定 3.11，与 README 一致）
- Gitee / GitLab 等其他 CI 平台

---

## 2. 方案选择

### 2.1 候选方案

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **A（采用）** | 单 Job + pip + `requirements.txt` | 最小改动、与现有依赖文件一致 | 无缓存时安装略慢 |
| B | 单 Job + uv | 与 README 推荐工具一致、安装更快 | 多一层工具；`pyproject.toml` 无 `[build-system]`，仍依赖 `requirements.txt` |
| C | 多 Job（test + lint） | lint 也纳入门禁 | 超出当前范围，增加维护成本 |

### 2.2 决策

采用 **方案 A**：单 workflow、单 job、pip 安装，Job 名固定为 `unit-tests` 供 Branch Protection 勾选。

---

## 3. Workflow 设计

### 3.1 文件

`.github/workflows/unit-tests.yml`

### 3.2 触发条件

```yaml
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
```

- **PR → main**：合并前必跑（门禁核心路径）
- **push → main**：合并后回归，防止 direct push 绕过 PR 时仍跑测

### 3.3 Job 定义

| 属性 | 值 |
|------|-----|
| workflow `name` | `CI` |
| job `id` | `unit-tests` |
| job `name` | `unit-tests` ← Branch Protection 勾选此名称 |
| runner | `ubuntu-latest` |
| Python | `3.11` |

### 3.4 步骤

1. `actions/checkout@v4`
2. `actions/setup-python@v5` — `python-version: "3.11"`
3. `pip install -r requirements.txt pytest pytest-asyncio`
4. `pytest tests/ -q`

> `pyproject.toml` 的 `[tool.pytest.ini_options]` 已限定 `python_files = "test_*.py"`，故 `pytest tests/` 等效于只跑单元测试，自动排除 `run_eval.py`。

### 3.5 CI 环境变量

| 变量 | 值 | 原因 |
|------|-----|------|
| `DASHSCOPE_API_KEY` | `ci-dummy-not-used` | `config.py` import 时 `_require_env` 必需 |
| `USE_MEMORY_CHECKPOINTER` | `true` | 避免测试间接依赖 Redis checkpoint |

可选（非必需）：`MCP_ENABLED=false` 会 skip `test_fetch_oncall` 中 1 个 MCP 集成用例；当前 mock 下保持默认 `true` 亦可全绿。

### 3.6 完整 workflow 草案

```yaml
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  unit-tests:
    name: unit-tests
    runs-on: ubuntu-latest
    env:
      DASHSCOPE_API_KEY: ci-dummy-not-used
      USE_MEMORY_CHECKPOINTER: "true"
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt pytest pytest-asyncio

      - name: Run unit tests
        run: pytest tests/ -q
```

### 3.7 预期耗时

| 阶段 | 估计 |
|------|------|
| 依赖安装 | ~1–2 min |
| pytest 124 用例 | ~30 s |
| **总计** | ~2–3 min |

---

## 4. Branch Protection 配置

Workflow 文件进仓库后，须在 GitHub 仓库设置中启用（无法完全代码化）：

**路径**: `Settings → Branches → Add branch protection rule`

| 配置项 | 值 |
|--------|-----|
| Branch name pattern | `main` |
| Require a pull request before merging | ✅ |
| Require status checks to pass before merging | ✅ |
| Required status checks | `unit-tests` |
| Require branches to be up to date before merging | ✅ |
| Do not allow bypassing the above settings | ✅（若可见） |

**首次启用步骤**:

1. 合并含 workflow 的 PR（或 push 到 `main`）
2. 开一条测试 PR 触发 CI，等待 `unit-tests` 跑完
3. 回到 Branch Protection，`unit-tests` 出现在可选 status check 列表
4. 勾选 `unit-tests` 为 required check 并保存

**失败行为**: PR 页显示 ❌，Merge 按钮禁用，直到修复后 CI 重跑通过。

---

## 5. 文档变更

在 `README.md` 增加 **「CI / 合并门禁」** 小节，内容：

- 本地跑法：`pytest tests/ -q`
- CI 触发条件与 scope 说明
- Branch Protection 手动配置步骤（§4 摘要）
- 说明无需 `.env` 即可跑单元测试（但 import `config` 时需要 `DASHSCOPE_API_KEY` 环境变量）

---

## 6. 验收标准

- [ ] PR → `main` 自动触发 `CI` workflow
- [ ] Job `unit-tests` 124 用例全部 PASS
- [ ] 故意改坏一个测试时 CI 失败
- [ ] Branch Protection 启用后，失败 PR 无法合并
- [ ] 本地 `pytest tests/ -q` 与 CI 命令一致
- [ ] 无 `.env`、无 GitHub Secrets 时 CI 可跑通

---

## 7. 实现清单

| 文件 | 动作 |
|------|------|
| `.github/workflows/unit-tests.yml` | **新建** |
| `README.md` | 追加 CI 小节 |
| GitHub Settings → Branches | 手动配置 Branch Protection |

---

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| `config.py` 新增 `_require_env` 字段导致 CI 缺 env 失败 | README/本 spec 文档化 dummy env；后续可考虑 `conftest.py` 统一 setenv |
| 测试变慢导致 CI 超时 | 当前 23s，远低于 GHA 6h 限制；未来可用 `pytest -m "not slow"` 分流 |
| status check 名称不匹配 | job `name` 固定 `unit-tests`，文档明确勾选名称 |
| 首次 workflow 未跑完无法勾选 check | 文档说明「先跑一条 PR 再配 Protection」 |
