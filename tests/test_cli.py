import copy
from pathlib import Path

import pytest
import yaml

from mrisk import cli

from test_compare import RULES, _checks, _db

ROOT = Path(__file__).resolve().parents[1]
DEMO_KEYS = [("database", "url"), ("data", "source"), ("data", "position_feed_dir"), ("snapshot_release",)]


def _drop(cfg: dict, path: tuple) -> None:
    for k in path[:-1]:
        cfg = cfg[k]
    cfg.pop(path[-1])


def test_demo_config_differs_from_main_config_only_in_demo_keys():
    main = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    demo = yaml.safe_load((ROOT / "config.demo.yaml").read_text(encoding="utf-8"))
    assert demo["data"]["source"] == "snapshot" and main["data"]["source"] == "akshare"
    assert demo["database"]["url"] != main["database"]["url"]
    assert demo["data"]["position_feed_dir"] != main["data"]["position_feed_dir"]
    a, b = copy.deepcopy(main), copy.deepcopy(demo)
    for path in DEMO_KEYS:
        _drop(b, path)
        if path != ("snapshot_release",):
            _drop(a, path)
    assert a == b


def test_dashboard_runs_from_config_dir_on_localhost(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, cwd, env: calls.append((cmd, cwd, env)) or 0)
    monkeypatch.setattr(cli, "get_engine", lambda url: pytest.fail("the dashboard command must not open a database"))
    monkeypatch.chdir(tmp_path)  # started from elsewhere: the defaults must still hold
    assert cli.main(["--config", str(ROOT / "config.demo.yaml"), "dashboard"]) == 0
    (cmd, cwd, env), = calls
    assert cmd[1:4] == ["-m", "streamlit", "run"] and Path(cmd[4]) == ROOT / "src" / "mrisk" / "dashboard" / "app.py"
    assert cmd[cmd.index("--server.address") + 1] == "localhost"
    assert cmd[cmd.index("--browser.gatherUsageStats") + 1] == "false"
    assert Path(cwd) == ROOT and Path(env["MRISK_CONFIG"]) == ROOT / "config.demo.yaml"


def test_dashboard_lan_address_only_when_asked(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.subprocess, "call", lambda cmd, cwd, env: calls.append(cmd) or 0)
    cli.main(["--config", str(ROOT / "config.demo.yaml"), "dashboard", "--address", "0.0.0.0", "--port", "8600"])
    cmd = calls[0]
    assert cmd[cmd.index("--server.address") + 1] == "0.0.0.0" and cmd[cmd.index("--server.port") + 1] == "8600"


def test_diff_db_exit_codes(tmp_path, capsys):
    _db(tmp_path / "a.db", RULES, _checks([0.5], ["2025-01-02"]))
    _db(tmp_path / "b.db", RULES, _checks([0.6], ["2025-01-02"]))
    _db(tmp_path / "c.db", RULES, _checks([0.5], ["2025-01-02"]))
    assert cli.main(["diff-db", str(tmp_path / "a.db"), str(tmp_path / "c.db")]) == 0
    assert "identical" in capsys.readouterr().out
    assert cli.main(["diff-db", str(tmp_path / "a.db"), str(tmp_path / "b.db")]) == 2
    assert cli.main(["diff-db", str(tmp_path / "a.db"), str(tmp_path / "missing.db")]) == 2
