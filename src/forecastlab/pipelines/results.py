"""Headline results across all phases -> reports/RESULTS.md

    python -m forecastlab.pipelines.results
"""
from __future__ import annotations

import json

import pandas as pd

from forecastlab.config import reports_dir


def load(stage: str) -> dict:
    p = reports_dir() / stage / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> None:
    lines = ["# Results", "", "Headline numbers from every phase. Each section links to the full report.", ""]

    def section(stage: str, title: str, body: list[str]) -> None:
        lines.extend([f"## [{title}]({stage}/README.md)", "", *body, ""])

    m = load("01_eda")
    if m:
        section("01_eda", "1. EDA", [f"* {m['n_series']:,} series x {m['n_days']:,} days; "
                                     f"{m['zero_share_last_year']:.0%} of item-days in the last year sell nothing.",
                                     f"* Weekend uplift vs Mon-Thu: {m['weekend_uplift']:+.0%}; SNAP lifts FOODS by "
                                     + ", ".join(f"{st} {m['snap_lift'][st]['FOODS']:+.0%}" for st in m['snap_lift'])
                                     + "."])
    m = load("02_data_quality")
    if m:
        section("02_data_quality", "2. Data quality", [
            f"* Suspected stock-outs: {m['stockout_share_of_on_shelf_days']:.2%} of on-shelf days; not-ranged runs: "
            f"{m.get('not_ranged_share', 0):.2%}.",
            "* Nightly gate on injected broken files: " + ", ".join(f"{k} -> {v}" for k, v in
                                                                m["validation_scenarios"].items()) + "."])
    m = load("04_validation")
    if m:
        ev = m["evaluations"]
        honest = ev["5. time split, direct 28-day forecast (honest)"]
        section("04_validation", "4. Validation & leakage", [
            "| evaluation | WAPE | vs honest |", "|---|---|---|",
            *[f"| {k} | {v:.1%} | {v / honest - 1:+.1%} |" for k, v in ev.items()]])
    m = load("05_baselines")
    if m:
        t = pd.DataFrame(m["wrmsse"]).T
        section("05_baselines", "5. Baselines (WRMSSE)", [t[["backtest mean"] + [c for c in t.columns if "holdout" in c]]
                                                      .sort_values("backtest mean").round(3).to_markdown()])
    m = load("14_backtest")
    if m:
        t = pd.DataFrame(m["ladder"]).set_index("model")
        cols = [c for c in ["rung", "fitted", "WRMSSE", "store x dept RMSSE", "item RMSSE", "WRMSSE (holdout)"]
                if c in t]
        section("14_backtest", "14. The ladder - does complex beat simple?", [
            f"Champion: **{m['champion']}** (runner-up {m['runner_up']}). Backtest mean of 3 origins; holdout once.", "",
            t[cols].round(3).to_markdown()])
    for stage, title, key in [("11_global_ml", "11. Global LightGBM design experiments", "experiments"),
                              ("12_deep_learning", "12. Deep learning", "results"),
                              ("13_foundation_models", "13. Foundation models (zero-shot)", "results")]:
        m = load(stage)
        if m and key in m:
            section(stage, title, [pd.DataFrame(m[key]).T.round(3).to_markdown()])
    m = load("15_probabilistic")
    if m:
        t = pd.DataFrame(m["item"]).set_index(["window", "method"])
        section("15_probabilistic", "15. Probabilistic forecasts (item level)", [t.round(3).to_markdown()])
    m = load("16_hierarchy")
    if m:
        t = pd.DataFrame(m["rmsse_by_level"]).T
        section("16_hierarchy", "16. Hierarchical reconciliation", [
            f"Best: **{m['best']}**.", "", t[["mean over levels", "holdout mean"]].round(3).to_markdown()])
    m = load("17_inventory")
    if m:
        t = pd.DataFrame(m["at_0.90"]).T
        section("17_inventory", "17. Inventory simulation (service quantile 0.90, 28 days)", [
            t[["fill rate", "avg inventory value $", "lost margin $", "total cost $"]].round(3).to_markdown()])
    m = load("21_champion_challenger")
    if m:
        section("21_champion_challenger", "21. Retraining / champion-challenger", [
            pd.DataFrame(m["candidates"]).T.round(4).to_markdown(), "",
            f"Promoted: {m['promoted'] or 'none'}; training/serving skew (max abs diff): {m['skew_max_abs']:.2e} units."])
    m = load("22_concept_drift")
    if m:
        section("22_concept_drift", "22. Concept drift", [
            pd.DataFrame(m["detection"]).set_index("signal").to_markdown(), "",
            pd.DataFrame(m["adaptation"]).T.round(3).to_markdown()])
    (reports_dir() / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
