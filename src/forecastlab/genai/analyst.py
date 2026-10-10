"""Phase 23 - GenAI forecast analyst.

    python -m forecastlab.genai.analyst CA_3 FOODS_3_090                     # grounded explanation (one call)
    python -m forecastlab.genai.analyst CA_3 FOODS_3_090 --ask "Should I order extra for Memorial Day?"
                                                                              # agentic: Claude picks the evidence

Planners do not read SHAP plots. The analyst turns the platform's evidence (forecast, history, calendar,
price plan, model drivers, monitoring) into a short explanation with risks and actions, in a fixed JSON shape.
Claude never computes the forecast - it explains the one the platform published. Without Anthropic credentials a
deterministic template explanation is produced from the same evidence.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import anthropic
from anthropic import beta_tool

from forecastlab.genai.evidence import EvidenceService, dumps

log = logging.getLogger(__name__)
MODEL = os.environ.get("FORECASTLAB_LLM_MODEL", "claude-opus-5-5")

SYSTEM = """You are a demand-planning analyst at a retailer. You explain the platform's published demand forecast
to store and category planners who decide orders and stock.

Ground every statement in the evidence provided: the published forecast and its P10/P90 range, recent sales,
the event/SNAP calendar, the planned price schedule, the model's driver factors, and monitoring results. Quote the
concrete numbers. Never invent a number or a cause that is not in the evidence; if something important is unknown
(e.g. promotions are not recorded in this data, competitor activity is unobserved), say so.

Driver factors are multiplicative effects from the model (1.15 = the driver raises the forecast by 15%); they explain
what the model used, not proven causes. Zero-sales runs may be stock-outs (censored demand), not lack of demand.
SNAP days are US food-assistance payment days that lift FOODS sales. Keep the summary short and plain - the reader is
a planner, not a data scientist. Your recommendations are advice; the planner decides."""

SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "description": "one sentence: what the forecast says"},
        "summary": {"type": "string", "description": "3-5 sentences explaining why, with numbers"},
        "drivers": {"type": "array", "items": {"type": "object", "properties": {
            "driver": {"type": "string"}, "direction": {"type": "string", "enum": ["up", "down", "neutral"]},
            "evidence": {"type": "string"}}, "required": ["driver", "direction", "evidence"],
            "additionalProperties": False}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "summary", "drivers", "risks", "confidence", "recommended_actions"],
    "additionalProperties": False,
}


def client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def llm_unavailable(e: Exception) -> bool:
    """No credentials or no network -> offline template. Anything else is a real error."""
    if isinstance(e, (anthropic.AuthenticationError, anthropic.APIConnectionError)):
        return True
    return isinstance(e, TypeError) and "Could not resolve authentication method" in str(e)


def llm_kwargs() -> dict:
    # fallbacks="default": if a safety classifier declines, the API re-runs the request on the recommended model.
    return {"model": MODEL, "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default",
            "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}}}


def parse(message) -> dict:
    if message.stop_reason == "refusal":
        raise RuntimeError(f"model declined: {getattr(message.stop_details, 'category', None)}")
    return json.loads(next(b.text for b in message.content if b.type == "text"))


def template(pack: dict) -> dict:
    """Offline fallback: the same evidence, fixed wording."""
    f, h, d, cal = pack["forecast"], pack["history"], pack["drivers"], pack["calendar"]
    chg = f["change_vs_last_28_days_pct"] or 0
    factors = d["average_factor_over_window"]
    top = sorted(factors.items(), key=lambda kv: abs((kv[1] or 1) - 1), reverse=True)[:3]
    drivers = [{"driver": k, "direction": "up" if v > 1.01 else "down" if v < 0.99 else "neutral",
                "evidence": f"average factor {v:.2f} over the next 28 days"} for k, v in top]
    risks = []
    if h["suspected_stockout_days"]:
        risks.append(f"{h['suspected_stockout_days']} suspected stock-out days in the last 8 weeks - recent sales "
                     "understate demand")
    if cal["events"]:
        risks.append("events in window: " + ", ".join(f"{e['event']} ({e['date']})" for e in cal["events"]))
    acc = pack["accuracy"]
    if acc.get("store_bias_last_28_days") is not None and abs(acc["store_bias_last_28_days"]) > 0.05:
        risks.append(f"store forecasts recently biased {acc['store_bias_last_28_days']:+.0%}")
    return {
        "headline": f"{f['series']}: {f['next_28_days_total']['forecast']:.0f} units expected over the next 28 days "
                    f"({chg:+.0f}% vs the last 28 days).",
        "summary": f"The last 28 days sold {f['last_28_days_actual_total']:.0f} units; the same period last year "
                   f"{f['same_28_days_last_year_actual_total']:.0f}. The model's strongest drivers are "
                   + ", ".join(f"{k} (x{v:.2f})" for k, v in top) + ".",
        "drivers": drivers, "risks": risks or ["no specific risks found in the evidence"],
        "confidence": "MEDIUM" if h["days_with_zero_sales"] < h["days"] / 2 else "LOW",
        "recommended_actions": ["Plan replenishment to the P90 daily values where shelf availability matters",
                                "Review again after the next nightly run"],
        "generated_by": "template (no Anthropic credentials available)",
    }


def explain(store_id: str, item_id: str, ev: EvidenceService | None = None) -> dict:
    ev = ev or EvidenceService()
    pack = ev.full_pack(store_id, item_id)
    try:
        msg = client().beta.messages.create(
            max_tokens=16000, system=SYSTEM,
            messages=[{"role": "user", "content": "Explain this forecast for the planner.\n\n"
                                                  f"<evidence>\n{dumps(pack)}\n</evidence>"}],
            **llm_kwargs())
        out = parse(msg)
        out["generated_by"] = f"{msg.model} (single call)"
        return out
    except Exception as e:
        if not llm_unavailable(e):
            raise
        log.warning("Claude unavailable (%s); using template", e)
        return template(pack)


def build_tools(ev: EvidenceService) -> list:
    def safe(fn, *a):
        try:
            return dumps(fn(*a))
        except KeyError as e:
            return json.dumps({"error": str(e)})

    @beta_tool
    def get_forecast(store_id: str, item_id: str) -> str:
        """Published 28-day forecast for one item in one store: daily mean, P10, P90, totals vs recent sales.

        Args:
            store_id: Store id, e.g. "CA_3".
            item_id: Item id, e.g. "FOODS_3_090".
        """
        return safe(ev.forecast, store_id, item_id)

    @beta_tool
    def get_sales_history(store_id: str, item_id: str) -> str:
        """Last 8 weeks of sales: weekly totals, weekday averages, zero-sales and suspected stock-out days.

        Args:
            store_id: Store id.
            item_id: Item id.
        """
        return safe(ev.sales_history, store_id, item_id)

    @beta_tool
    def get_calendar() -> str:
        """Events/holidays and SNAP days (per state) inside the forecast window."""
        return dumps(ev.calendar())

    @beta_tool
    def get_price_plan(store_id: str, item_id: str) -> str:
        """Current price, recent maximum, and planned price changes in the forecast window.

        Args:
            store_id: Store id.
            item_id: Item id.
        """
        return safe(ev.price_plan, store_id, item_id)

    @beta_tool
    def get_model_drivers(store_id: str, item_id: str) -> str:
        """Model driver factors (TreeSHAP, multiplicative) for the item's forecast: weekday, events, SNAP, price,
        recent sales level, season.

        Args:
            store_id: Store id.
            item_id: Item id.
        """
        return safe(ev.drivers, store_id, item_id)

    @beta_tool
    def get_accuracy(store_id: str, item_id: str = "") -> str:
        """Recent live accuracy and bias for the store (and the item's recent 1-day-ahead errors) plus alerts.

        Args:
            store_id: Store id.
            item_id: Optional item id.
        """
        return safe(ev.accuracy, store_id, item_id or None)

    @beta_tool
    def get_aggregate_forecast(level: str, key: str) -> str:
        """Published forecast for an aggregate, e.g. level "store" key "CA_3", level "store x department"
        key "CA_3/FOODS_3", level "department" key "FOODS_3", level "total" key "total".

        Args:
            level: One of total, state, store, category, department, state x category, state x department,
                store x category, store x department.
            key: The aggregate key.
        """
        return safe(ev.aggregate, level, key)

    return [get_forecast, get_sales_history, get_calendar, get_price_plan, get_model_drivers, get_accuracy,
            get_aggregate_forecast]


def ask(question: str, store_id: str, item_id: str, ev: EvidenceService | None = None,
        max_iterations: int = 10) -> dict:
    """Agentic mode: Claude decides which evidence to fetch to answer a planner's question."""
    ev = ev or EvidenceService()
    try:
        runner = client().beta.messages.tool_runner(
            max_tokens=16000, system=SYSTEM + "\n\nUse the tools to gather only the evidence the question needs, "
                                              "then answer in the required JSON shape.",
            tools=build_tools(ev), max_iterations=max_iterations,
            messages=[{"role": "user", "content": f"Item {item_id} in store {store_id}. Planner question: {question}"}],
            **llm_kwargs())
        trace, final = [], None
        for message in runner:
            for block in message.content:
                if block.type == "tool_use":
                    trace.append({"tool": block.name, "input": block.input})
            final = message
        out = parse(final)
        out["tool_calls"] = trace
        out["generated_by"] = f"{final.model} (agent, {len(trace)} tool calls)"
        return out
    except Exception as e:
        if not llm_unavailable(e):
            raise
        log.warning("Claude unavailable (%s); using template", e)
        return template(ev.full_pack(store_id, item_id))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("store_id")
    ap.add_argument("item_id")
    ap.add_argument("--ask", default=None, help="planner question -> agentic mode")
    a = ap.parse_args()
    out = ask(a.ask, a.store_id, a.item_id) if a.ask else explain(a.store_id, a.item_id)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
