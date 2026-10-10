"""Shared chart style for the static PNG figures in reports/.

Palette: categorical slots in fixed order (never cycled), single-hue blue for
magnitude, recessive grid/axes, one y-axis per chart.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
})


def save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


def bar(ax, labels, values, color=SERIES[0], fmt="{:.1%}", horizontal=False):
    """Thin bars with a surface gap between neighbours and selective value labels."""
    if horizontal:
        bars = ax.barh(labels, values, color=color, height=0.7, edgecolor=SURFACE, linewidth=2)
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    else:
        bars = ax.bar(labels, values, color=color, width=0.7, edgecolor=SURFACE, linewidth=2)
    if fmt and len(values) <= 12:
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], color=INK_2, fontsize=8, padding=2)
    return bars
