"""Phase 23 - GenAI forecast analyst: examples written into a report.

    python -m forecastlab.pipelines.genai          (after `make nightly`)
"""
from __future__ import annotations

import json
import logging

import numpy as np

from forecastlab.genai.analyst import MODEL, ask, explain
from forecastlab.genai.evidence import EvidenceService, dumps
from forecastlab.report import Report

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ev = EvidenceService()
    r = Report("23_genai_analyst", "Phase 23 - GenAI Forecast Analyst")
    ds = ev.ds
    rows = np.flatnonzero(ds.series["store_id"].to_numpy() == "CA_3")
    vol = ev.Y[rows, ev.origin - 27: ev.origin + 1].sum(axis=1)
    top = ds.series.iloc[rows[np.argsort(-vol)[:1]]]["item_id"].iloc[0]
    examples = [("CA_3", top, None),
                ("TX_2", "FOODS_3_090", "Should I order extra for the events in the next four weeks?")]
    r.p(f"""Run `{ev.run['run_id']}` (data through {ev.run['origin_date']}). The analyst receives only platform evidence
(`genai/evidence.py`): published forecast with P10/P90, recent sales and suspected stock-outs, events and SNAP days,
the planned price schedule, SHAP driver factors and monitoring results. Model: `{MODEL}` with structured JSON output
and server-side refusal fallback. Without credentials the platform falls back to a deterministic template built from
the same evidence, so the feature degrades instead of failing.""")
    for store, item, q in examples:
        out = ask(q, store, item, ev) if q else explain(store, item, ev)
        r.h2(f"{item} @ {store}" + (f" - question: \"{q}\"" if q else " - explain the forecast"))
        r.code(json.dumps(out, indent=2), "json")
    pack = ev.full_pack("CA_3", top)
    r.h2("Evidence pack (what the model sees), abbreviated")
    pack["forecast"]["daily"] = pack["forecast"]["daily"][:3] + [{"...": "25 more days"}]
    r.code(dumps(pack)[:4000], "json")
    r.p("""**Design rules**
* The LLM explains; it never forecasts. Numbers come from the platform and are quoted, not generated.
* Fixed JSON schema (`output_config.format`) -> the dashboard renders it and tests can validate it.
* Agentic mode (a planner question) lets Claude choose tools (`get_forecast`, `get_calendar`, `get_price_plan`,
  `get_model_drivers`, `get_accuracy`, `get_aggregate_forecast`...) via the SDK tool runner; the trace of tool calls is
  returned for audit.
* Uncertainty and data caveats (stock-outs, no promotion data) are part of the prompt, so explanations carry them.""")
    r.save()


if __name__ == "__main__":
    main()
