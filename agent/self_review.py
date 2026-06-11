"""
Self-review loop: at the end of every cycle, Claude critiques the run's outputs
and the script's current configuration, then applies bounded tuning adjustments.

What it reviews each run:
- Output quality: factor count, narrative coverage, signal distribution sanity
- Calibration: backtest accuracy vs signal thresholds
- Configuration drift: category weights, thresholds, search queries

What it can change automatically (within safe bounds):
- BUY/SELL thresholds (0.20–0.60 absolute)
- Category weights (0.5–2.0)

Everything else (code-level suggestions) is logged to the self_reviews table
and printed for a human to act on.
"""

import json
import os
import re
from datetime import datetime

import anthropic
from rich.console import Console
from rich.panel import Panel

import config
import agent.state_store as state_store

console = Console()

TUNING_PATH = os.path.join(os.path.dirname(config.DB_PATH), "tuning.json")

REVIEW_SYSTEM = """You are a quantitative engineering reviewer auditing an automated
semiconductor trading-signal agent after one of its runs. You critique the run's
outputs and configuration, and propose conservative tuning.

Be specific and skeptical. A run with all-BUY signals, zero factors, missing
narratives, or accuracy below 50% deserves a low score and concrete fixes.

Output ONLY valid JSON — no markdown, no explanation.
"""


def load_persisted_tuning() -> dict:
    """Apply previously saved tuning at the start of a run. Returns what was applied."""
    if not os.path.exists(TUNING_PATH):
        return {}
    try:
        with open(TUNING_PATH) as fh:
            tuning = json.load(fh)
    except Exception:
        return {}
    _apply_tuning(tuning)
    return tuning


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _apply_tuning(tuning: dict):
    if "buy_threshold" in tuning:
        config.BUY_THRESHOLD = _clamp(tuning["buy_threshold"], 0.20, 0.60)
    if "sell_threshold" in tuning:
        config.SELL_THRESHOLD = _clamp(tuning["sell_threshold"], -0.60, -0.20)
    for cat, w in (tuning.get("category_weights") or {}).items():
        if cat in config.CATEGORY_WEIGHTS:
            config.CATEGORY_WEIGHTS[cat] = _clamp(w, 0.5, 2.0)


def _persist_tuning(tuning: dict):
    os.makedirs(os.path.dirname(TUNING_PATH), exist_ok=True)
    existing = {}
    if os.path.exists(TUNING_PATH):
        try:
            with open(TUNING_PATH) as fh:
                existing = json.load(fh)
        except Exception:
            pass
    merged_weights = {**(existing.get("category_weights") or {}), **(tuning.get("category_weights") or {})}
    existing.update({k: v for k, v in tuning.items() if k != "category_weights"})
    if merged_weights:
        existing["category_weights"] = merged_weights
    existing["updated_at"] = datetime.utcnow().isoformat()
    with open(TUNING_PATH, "w") as fh:
        json.dump(existing, fh, indent=2)


def build_cycle_summary(
    snippets_count: int,
    used_fallback: bool,
    factors: list[dict],
    forward_factors: list[dict],
    recommendations: list[dict],
    accuracy_stats: dict,
    errors: list[str],
) -> dict:
    signal_dist: dict[str, int] = {}
    for r in recommendations:
        signal_dist[r["signal"]] = signal_dist.get(r["signal"], 0) + 1

    narratives = sum(1 for f in factors if f.get("narrative"))
    dated = sum(1 for f in factors if f.get("event_date"))

    return {
        "snippets_collected": snippets_count,
        "used_knowledge_fallback": used_fallback,
        "factors_extracted": len(factors),
        "factors_with_event_date": dated,
        "factors_with_narrative": narratives,
        "forward_factors": len(forward_factors),
        "signal_distribution": signal_dist,
        "backtest": accuracy_stats or {},
        "runtime_errors": errors,
        "current_config": {
            "buy_threshold": config.BUY_THRESHOLD,
            "sell_threshold": config.SELL_THRESHOLD,
            "category_weights": dict(config.CATEGORY_WEIGHTS),
        },
    }


def run_self_review(cycle_summary: dict) -> dict:
    """Critique this run and apply bounded tuning. Returns the review dict."""
    console.print("  [dim]Self-review: auditing this run's outputs and config…[/dim]")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=1500,
        temperature=0.2,
        system=REVIEW_SYSTEM,
        messages=[{"role": "user", "content": f"""
=== THIS RUN'S TELEMETRY ===
{json.dumps(cycle_summary, indent=2)}

Audit this run. Consider:
- Were enough factors extracted, with dates and narratives attached?
- Is the signal distribution plausible, or suspiciously one-sided?
- Does backtest accuracy justify the current BUY/SELL thresholds?
  (low BUY accuracy → raise buy_threshold; signals all HOLD → consider lowering)
- Are category weights consistent with which factor types actually moved prices?
- Any runtime errors that degrade output quality?

Return JSON:
{{
  "output_quality_score": <0-10>,
  "critique": "<3-5 sentences: what was good, what was weak, what degraded quality>",
  "tuning": {{
    "buy_threshold": <float or omit>,
    "sell_threshold": <float or omit>,
    "category_weights": {{"<category>": <float>, ...}} or omit
  }},
  "code_suggestions": ["<concrete improvement a developer should make>", ...]
}}

Only include tuning keys that should change. Empty tuning is fine if calibration looks right.
"""}],
    )

    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    review = json.loads(raw)

    tuning = review.get("tuning") or {}
    if tuning:
        _apply_tuning(tuning)
        _persist_tuning(tuning)
        console.print(f"  [cyan]  → Tuning applied: {json.dumps(tuning)}[/cyan]")

    state_store.save_self_review(
        score=review.get("output_quality_score"),
        critique=review.get("critique", ""),
        tuning=tuning,
        code_suggestions=review.get("code_suggestions", []),
    )

    score = review.get("output_quality_score", "?")
    color = "green" if isinstance(score, (int, float)) and score >= 7 else "yellow" if isinstance(score, (int, float)) and score >= 4 else "red"
    body = f"[bold {color}]Quality score: {score}/10[/bold {color}]\n\n{review.get('critique', '')}"
    suggestions = review.get("code_suggestions") or []
    if suggestions:
        body += "\n\n[bold]Suggested improvements:[/bold]"
        for s in suggestions[:5]:
            body += f"\n  • {s}"
    console.print(Panel(body, title="Self-Review", border_style=color))

    return review
