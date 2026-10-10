"""Config loading and well-known paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("FORECASTLAB_ROOT", Path(__file__).resolve().parents[2]))


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    path = Path(os.environ.get("FORECASTLAB_CONFIG", ROOT / "configs" / "config.yaml"))
    with open(path) as f:
        return yaml.safe_load(f)


def _dir(key: str) -> Path:
    p = ROOT / load_config()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_dir() -> Path:
    return ROOT / load_config()["paths"]["raw_dir"]


def artifacts_dir() -> Path:
    return _dir("artifacts_dir")


def reports_dir() -> Path:
    return _dir("reports_dir")
