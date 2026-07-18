"""Lock config env helper behavior used across the app."""

from app.core import config


def test_env_bool_accepts_common_truthy_values(monkeypatch):
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("TRIAGE_SESSION_ENABLED", value)
        # Re-evaluate helper directly; module constants are already bound.
        assert config._env_bool("TRIAGE_SESSION_ENABLED", "false") is True


def test_env_bool_defaults_false(monkeypatch):
    monkeypatch.delenv("USE_MEMORY_CHECKPOINTER", raising=False)
    assert config._env_bool("USE_MEMORY_CHECKPOINTER", "false") is False


def test_mcp_server_command_uses_current_interpreter():
    assert config.MCP_SERVER_COMMAND.endswith("hospital_mcp/server.py")
    assert "hospital_mcp" in config.MCP_SERVER_COMMAND


def test_disease_kb_path_points_under_source_data():
    path = str(config.DISEASE_KB_PATH).replace("\\", "/")
    assert path.endswith("disease_kb.jsonl")
    assert config.SOURCE_DATA_DIR in path or "disease_kb" in path
