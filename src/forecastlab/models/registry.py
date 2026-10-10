"""Model registry: which forecaster is champion, which is challenger, and the evidence for each.

    artifacts/models/registry.json
    artifacts/models/<version>/<store>.txt     trained LightGBM boosters (production versions)

A registry entry records *how* the model is trained (spec), on which data window, and its backtest / holdout /
live metrics, so a promotion decision can be audited later.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from forecastlab.config import artifacts_dir


def models_dir() -> Path:
    d = artifacts_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def registry_path() -> Path:
    return models_dir() / "registry.json"


def load_registry() -> dict:
    p = registry_path()
    return json.loads(p.read_text()) if p.exists() else {"models": {}, "champion": None, "challenger": None,
                                                         "history": []}


def save_registry(reg: dict) -> None:
    registry_path().write_text(json.dumps(reg, indent=2, default=float))


def register(version: str, spec: dict, metrics: dict, role: str | None = None, **extra) -> dict:
    reg = load_registry()
    entry = reg["models"].get(version, {})
    entry.update({"spec": spec, "metrics": {**entry.get("metrics", {}), **metrics},
                  "registered_at": entry.get("registered_at", time.strftime("%Y-%m-%dT%H:%M:%S")), **extra})
    reg["models"][version] = entry
    if role:
        set_role(reg, version, role, reason=extra.get("reason", "registered"))
    save_registry(reg)
    return reg


def set_role(reg: dict, version: str, role: str, reason: str) -> None:
    previous = reg.get(role)
    reg[role] = version
    reg.setdefault("history", []).append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "version": version,
                                          "previous": previous, "reason": reason})


def promote(version: str, reason: str) -> dict:
    """Challenger -> champion; the old champion is kept as an archived version (instant rollback)."""
    reg = load_registry()
    old = reg.get("champion")
    set_role(reg, version, "champion", reason)
    if reg.get("challenger") == version:
        reg["challenger"] = None
    if old and old != version:
        reg["models"][old]["status"] = "archived"
    reg["models"][version]["status"] = "champion"
    save_registry(reg)
    return reg


def version_dir(version: str) -> Path:
    d = models_dir() / version
    d.mkdir(parents=True, exist_ok=True)
    return d
