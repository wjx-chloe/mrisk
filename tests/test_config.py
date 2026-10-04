from mrisk.config import database_url, load_config
from mrisk.jobs.gate import blocking_scope


def test_repo_config_loads_both_confidence_levels():
    cfg = load_config("config.yaml")
    assert cfg["confidence"] == [0.95, 0.99]
    assert cfg["lookback_days"] == 250


def test_sqlite_path_resolved_relative_to_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("database:\n  url: sqlite:///sub/x.db\n", encoding="utf-8")
    url = database_url(load_config(p))
    assert url == "sqlite:///" + (tmp_path / "sub" / "x.db").as_posix()
    assert (tmp_path / "sub").is_dir()


def test_repo_config_classifies_blocking_rules_by_how_long_the_data_is_used():
    assert blocking_scope(load_config("config.yaml")) == {
        "price": ["DQ02", "DQ03", "DQ11"],
        "position": ["DQ07"],
        "static": ["DQ08", "DQ09"],
    }
