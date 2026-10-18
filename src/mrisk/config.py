import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path or os.environ.get("MRISK_CONFIG", "config.yaml")).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = path.parent
    cfg["_path"] = path
    return cfg


def resolve_path(cfg: dict[str, Any], p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else cfg["_root"] / path


def database_url(cfg: dict[str, Any]) -> str:
    url = cfg["database"]["url"]
    prefix = "sqlite:///"
    if url.startswith(prefix) and url != "sqlite:///:memory:":
        db_path = resolve_path(cfg, url[len(prefix):])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return prefix + db_path.as_posix()
    return url
