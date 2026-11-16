from types import SimpleNamespace

import sf_framework.cli as cli
import sf_framework.runner as runner


def test_api_login_uses_stdin_and_delegates_to_codex(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("SF_TEST_LOGIN_KEY", "test-secret")
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert (
        cli.main(["auth", "login", "--method", "api-key", "--api-key-env", "SF_TEST_LOGIN_KEY"])
        == 0
    )
    assert captured["command"] == ["codex", "login", "--with-api-key"]
    assert captured["input"] == "test-secret\n"
    assert "test-secret" not in " ".join(captured["command"])


def test_chat_override_records_api_auth_even_from_account_config(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[agent]\nkind="codex"\nauth="chatgpt"\n')
    captured = {}

    def fake_run(config, **kwargs):
        captured["config"] = config
        return {"agent_statuses": {}, "results": []}

    monkeypatch.setattr(runner, "run", fake_run)
    assert (
        cli.main(["run", "--config", str(path), "--agent", "openai_chat", "--model", "non-gpt"])
        == 0
    )
    assert captured["config"].agent.auth == "api_key"
    assert captured["config"].agent.model == "non-gpt"


def test_workers_cli_overrides_toml(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text("[run]\nworkers=2\n")
    captured = {}

    def fake_run(config, **kwargs):
        captured["workers"] = config.run.workers
        return {"agent_statuses": {}, "results": []}

    monkeypatch.setattr(runner, "run", fake_run)
    assert cli.main(["run", "--config", str(path), "--workers", "4"]) == 0
    assert captured["workers"] == 4
