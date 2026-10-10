"""Tiny Markdown report builder: every pipeline stage writes reports/<stage>/README.md."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from forecastlab.config import reports_dir


class Report:
    def __init__(self, stage: str, title: str) -> None:
        self.dir: Path = reports_dir() / stage
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lines: list[str] = [f"# {title}", ""]
        self.metrics: dict = {}

    def h2(self, text: str) -> None:
        self.lines += [f"## {text}", ""]

    def p(self, text: str) -> None:
        self.lines += [text.strip(), ""]

    def table(self, df: pd.DataFrame, floatfmt: str = ".4f", index: bool = True) -> None:
        # Whole-number columns (counts) print as integers with thousands separators.
        fmts = [",.0f" if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().all() and (df[c] % 1 == 0).all()
                else floatfmt for c in df.columns]
        if index:
            fmts = [floatfmt] + fmts
        self.lines += [df.to_markdown(floatfmt=fmts, index=index), ""]

    def code(self, text: str, lang: str = "text") -> None:
        self.lines += [f"```{lang}", text.rstrip(), "```", ""]

    def figure(self, filename: str, alt: str) -> None:
        self.lines += [f"![{alt}]({filename})", ""]

    def save(self) -> Path:
        out = self.dir / "README.md"
        out.write_text("\n".join(self.lines))
        if self.metrics:
            (self.dir / "metrics.json").write_text(json.dumps(self.metrics, indent=2, default=float))
        return out
