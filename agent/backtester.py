"""
Backtester: compares past BUY/HOLD/SELL signals against subsequent price moves,
identifies systematic discrepancies, and asks Claude to adjust category weights.
"""

import json
from datetime import datetime
from typing import Optional

import anthropic
from rich.console import Console
from rich.table import Table
from rich import box

import config
import agent.state_store as state_store

console = Console()

BACKTESTER_SYSTEM = """You are a quantitative analyst reviewing the accuracy of a
semiconductor factor-analysis model. Your job is to diagnose systematic errors and
recommend weight adjustments to improve future signal accuracy.

Output ONLY valid JSON — no markdown, no explanation.
"""


def _direction_match(signal: str, pct_change: float) -> str:
    """
    Returns: 'correct', 'incorrect', or 'neutral'
    BUY correct if price went up, SELL correct if price went down.
    HOLD is correct if price stayed within ±2%.
    """
    if signal == "BUY":
        return "correct" if pct_change > 1.0 else ("neutral" if pct_change >= -1.0 else "incorrect")
    elif signal == "SELL":
        return "correct" if pct_change < -1.0 else ("neutral" if pct_change <= 1.0 else "incorrect")
    else:  # HOLD
        return "correct" if abs(pct_change) <= 2.0 else ("neutral" if abs(pct_change) <= 4.0 else "incorrect")


def compute_outcomes(price_data_current: dict) -> list[dict]:
    """
    For each previous run pair, compute whether the signal matched the next price move.
    Uses price_14d_change from consecutive recommendations as a proxy for move direction.
    """
    pairs = state_store.get_paired_runs()
    if not pairs:
        return []

    new_outcomes = []
    already_scored = {
        (o["run_id"], o["ticker"])
        for o in state_store.get_all_signal_outcomes(limit=10000)
    }

    for prev_run, next_run in pairs:
        prev_recs = state_store.get_recommendations_for_run(prev_run["run_id"])
        next_recs = {
            r["ticker"]: r
            for r in state_store.get_recommendations_for_run(next_run["run_id"])
        }

        for rec in prev_recs:
            ticker = rec["ticker"]
            key = (prev_run["run_id"], ticker)
            if key in already_scored:
                continue

            prev_price_chg = rec.get("price_14d_change") or 0.0
            next_rec = next_recs.get(ticker)
            if not next_rec:
                continue
            next_price_chg = next_rec.get("price_14d_change") or 0.0

            # Approximate inter-run move as the delta in 14d price change
            approx_move = next_price_chg - prev_price_chg

            outcome = _direction_match(rec["signal"], approx_move)
            new_outcomes.append({
                "run_id": prev_run["run_id"],
                "ticker": ticker,
                "signal": rec["signal"],
                "composite_score": rec.get("composite_score"),
                "price_at_signal": prev_price_chg,
                "price_at_next_run": next_price_chg,
                "actual_pct_change": round(approx_move, 2),
                "outcome": outcome,
            })

    if new_outcomes:
        state_store.save_signal_outcomes(new_outcomes)

    return new_outcomes


def compute_accuracy_stats(outcomes: list[dict]) -> dict:
    all_outcomes = state_store.get_all_signal_outcomes(limit=500)
    if not all_outcomes:
        return {}

    stats: dict[str, dict] = {}
    for o in all_outcomes:
        sig = o["signal"]
        stats.setdefault(sig, {"correct": 0, "incorrect": 0, "neutral": 0, "total": 0})
        stats[sig][o["outcome"]] += 1
        stats[sig]["total"] += 1

    overall_correct = sum(s["correct"] for s in stats.values())
    overall_total = sum(s["total"] for s in stats.values())
    accuracy_pct = round(overall_correct / overall_total * 100, 1) if overall_total else 0.0

    return {"by_signal": stats, "overall_accuracy_pct": accuracy_pct, "total_evaluated": overall_total}


def _ask_claude_for_adjustments(stats: dict, recent_outcomes: list[dict]) -> tuple[dict, str]:
    """Ask Claude to diagnose errors and recommend weight adjustments."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    recent_text = json.dumps(recent_outcomes[-30:], indent=2)
    stats_text = json.dumps(stats, indent=2)
    current_weights = json.dumps(config.CATEGORY_WEIGHTS, indent=2)

    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=2048,
        temperature=0.2,
        system=BACKTESTER_SYSTEM,
        messages=[{"role": "user", "content": f"""
=== CURRENT CATEGORY WEIGHTS ===
{current_weights}

=== SIGNAL ACCURACY STATS ===
{stats_text}

=== RECENT SIGNAL OUTCOMES (signal vs actual price move) ===
{recent_text}

Analyze the discrepancies between signals and actual price moves.
Identify which signal types (BUY/HOLD/SELL) are systematically wrong.
Consider whether category weights need rebalancing.

Return a JSON object with two keys:
{{
  "weight_adjustments": {{
    "<category_name>": <new_weight_float>,
    ...
  }},
  "reasoning": "<2-3 sentence explanation of what was wrong and why these adjustments fix it>"
}}

Only include categories that need adjustment. Weights should stay in range 0.5–2.0.
If accuracy is already good (>65%) or sample size is too small (<5 outcomes),
return empty weight_adjustments and say so in reasoning.
"""}],
    )

    raw = resp.content[0].text.strip()
    raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
    data = json.loads(raw)
    return data.get("weight_adjustments", {}), data.get("reasoning", "")


def run_backtest(price_data_current: dict) -> tuple[dict, dict]:
    """
    Full backtest cycle. Returns (accuracy_stats, weight_adjustments).
    Weight adjustments are applied to config at runtime for this cycle.
    """
    console.print("  [dim]Running backtest against past signals…[/dim]")

    new_outcomes = compute_outcomes(price_data_current)
    stats = compute_accuracy_stats(new_outcomes)

    if not stats or stats.get("total_evaluated", 0) < 2:
        console.print("  [dim]  → Not enough history to backtest yet[/dim]")
        # Load any previously computed adjustments
        saved = state_store.get_latest_weight_adjustments()
        return stats, saved

    accuracy = stats.get("overall_accuracy_pct", 0)
    console.print(
        f"  [dim]  → {stats['total_evaluated']} outcomes evaluated, "
        f"accuracy: [{'green' if accuracy >= 60 else 'yellow' if accuracy >= 40 else 'red'}]{accuracy}%[/][/dim]"
    )

    # Ask Claude for weight adjustments
    all_outcomes = state_store.get_all_signal_outcomes(limit=100)
    try:
        adjustments, reasoning = _ask_claude_for_adjustments(stats, all_outcomes)
    except Exception as e:
        console.print(f"  [yellow]  → Weight adjustment skipped: {e}[/yellow]")
        adjustments, reasoning = {}, ""

    if adjustments:
        console.print(f"  [cyan]  → Applying weight adjustments: {adjustments}[/cyan]")
        console.print(f"  [dim]    Reason: {reasoning[:120]}[/dim]")
        state_store.save_weight_adjustment(adjustments, reasoning, accuracy)
        # Apply to runtime config
        for cat, w in adjustments.items():
            if cat in config.CATEGORY_WEIGHTS:
                config.CATEGORY_WEIGHTS[cat] = float(w)
    else:
        console.print("  [dim]  → No weight adjustments needed[/dim]")
        # Still apply any previously saved adjustments
        saved = state_store.get_latest_weight_adjustments()
        for cat, w in saved.items():
            if cat in config.CATEGORY_WEIGHTS:
                config.CATEGORY_WEIGHTS[cat] = float(w)

    return stats, adjustments


def print_accuracy_table(stats: dict):
    if not stats:
        return
    table = Table(title="Signal Accuracy (past cycles)", box=box.SIMPLE)
    table.add_column("Signal", style="bold")
    table.add_column("Correct", style="green")
    table.add_column("Incorrect", style="red")
    table.add_column("Neutral", style="yellow")
    table.add_column("Accuracy %")

    for sig, s in stats.get("by_signal", {}).items():
        total = s["total"] or 1
        acc = round(s["correct"] / total * 100, 1)
        table.add_row(sig, str(s["correct"]), str(s["incorrect"]), str(s["neutral"]), f"{acc}%")

    overall = stats.get("overall_accuracy_pct", 0)
    color = "green" if overall >= 60 else "yellow" if overall >= 40 else "red"
    table.add_row(
        "[bold]TOTAL[/bold]",
        "", "", "",
        f"[bold {color}]{overall}%[/bold {color}]",
    )
    console.print(table)
