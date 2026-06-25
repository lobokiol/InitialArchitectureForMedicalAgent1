# 归档脚本

从 `scripts/` 移入的手动测试脚本，不参与生产启动路径。

运行示例（在项目根目录）：

```powershell
.\.venv\Scripts\python.exe scripts/archive/test_chat_api.py --base-url http://127.0.0.1:8000
.\.venv\Scripts\python.exe scripts/archive/test_decision_routing.py
```

`dev-services.ps1 -Action verify` 在 `ChatApiTests=true` 时会调用 `test_chat_api.py`。
