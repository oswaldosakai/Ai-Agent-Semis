"""CLI and HTML report generation."""

import os
import json
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
from jinja2 import Environment, FileSystemLoader, select_autoescape

import config

console = Console()


def _signal_style(signal: str) -> str:
    return {"BUY": "bold green", "SELL": "bold red", "HOLD": "yellow"}.get(signal, "white")


def print_cli_report(recommendations: list[dict], factors: list[dict], run_ts: str,
                     data_note: Optional[str] = None):
    console.rule(f"[bold cyan]Semis Factor Analysis — {run_ts}[/bold cyan]")
    if data_note:
        console.print(f"  [yellow]{data_note}[/yellow]\n")

    # ── Signals table ────────────────────────────────────────────────────────
    table = Table(title="Signals", box=box.ROUNDED, show_lines=True)
    table.add_column("Ticker", style="bold white", width=8)
    table.add_column("Signal", width=6)
    table.add_column("Score", width=7)
    table.add_column("14d %", width=8)
    table.add_column("Top Factors", min_width=40)

    for r in sorted(recommendations, key=lambda x: x["composite_score"], reverse=True):
        sig = r["signal"]
        pct = r.get("price_14d_change", 0) or 0
        pct_str = f"{pct:+.1f}%"
        def _fmt_factor(tf):
            if isinstance(tf, dict):
                d = f" [{tf['event_date']}]" if tf.get("event_date") else ""
                return tf.get("description", "")[:80] + d
            return str(tf)[:80]

        table.add_row(
            r["ticker"],
            Text(sig, style=_signal_style(sig)),
            f"{r['composite_score']:+.3f}",
            pct_str,
            "; ".join(_fmt_factor(tf) for tf in r.get("top_factors", [])[:2]),
        )
    console.print(table)

    # ── Last 48 Hours ────────────────────────────────────────────────────────
    recent = _get_recent_48h_factors(factors)
    if recent:
        console.rule("[bold yellow]Last 48 Hours — Most Impactful[/bold yellow]")
        for f in recent:
            sentiment_icon = {"positive": "↑", "negative": "↓", "neutral": "→"}.get(f.get("sentiment"), "")
            color = {"positive": "green", "negative": "red", "neutral": "white"}.get(f.get("sentiment"), "white")
            date_str = f.get("event_date") or f"~{f.get('recency_days', 0):.0f}d ago"
            console.print(
                f"  [{color}]{sentiment_icon} [{f.get('category','?')}][/{color}] "
                f"[dim]{date_str}[/dim] — {f.get('description','')[:110]}"
            )

    # ── Top factors with deep-dive narratives ────────────────────────────────
    console.rule("[bold]Key Factors This Cycle[/bold]")
    top = sorted(factors, key=lambda x: abs(x.get("raw_score", 0)), reverse=True)[:8]
    for i, f in enumerate(top):
        sentiment_icon = {"positive": "↑", "negative": "↓", "neutral": "→"}.get(f.get("sentiment"), "")
        color = {"positive": "green", "negative": "red", "neutral": "white"}.get(f.get("sentiment"), "white")
        console.print(
            f"  [{color}]{sentiment_icon} [{f.get('category','?')}][/{color}] "
            f"{f.get('description','')[:120]} "
            f"[dim](mag={f.get('magnitude')}, {f.get('recency_days',0):.0f}d ago)[/dim]"
        )
        if i < 3 and f.get("narrative"):
            # Wrap narrative text at 100 chars for clean terminal display
            narrative = f["narrative"]
            for line in [narrative[j:j+100] for j in range(0, min(len(narrative), 300), 100)]:
                console.print(f"    [dim]{line}[/dim]")


def _load_template():
    template_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html"]),
    )
    return env.get_template("report.html")


def _get_recent_48h_factors(factors: list[dict]) -> list[dict]:
    """Return up to 5 most impactful factors from the last 48 hours."""
    recent = [f for f in factors if (f.get("recency_days") or 99) <= 2]
    return sorted(recent, key=lambda x: abs(x.get("raw_score", 0)), reverse=True)[:5]


def generate_html_report(
    recommendations: list[dict],
    factors: list[dict],
    forward_factors: list[dict],
    run_ts: str,
    historical_recs: list[dict],
    data_estimated: bool = False,
    accuracy_stats: Optional[dict] = None,
    weight_adjustments: Optional[dict] = None,
    self_review: Optional[dict] = None,
) -> str:
    os.makedirs(config.REPORT_OUTPUT_DIR, exist_ok=True)
    filename = f"report_{run_ts.replace(':', '-').replace(' ', '_')}.html"
    filepath = os.path.join(config.REPORT_OUTPUT_DIR, filename)

    sorted_factors = sorted(factors, key=lambda x: abs(x.get("raw_score", 0)), reverse=True)
    recent_48h = _get_recent_48h_factors(factors)

    template = _load_template()
    html = template.render(
        run_ts=run_ts,
        model=config.CLAUDE_MODEL,
        recommendations=recommendations,
        factors=sorted_factors,
        forward_factors=forward_factors,
        historical_recs=historical_recs,
        buy_threshold=config.BUY_THRESHOLD,
        sell_threshold=config.SELL_THRESHOLD,
        data_estimated=data_estimated,
        accuracy_stats=accuracy_stats or {},
        weight_adjustments=weight_adjustments or {},
        category_weights=config.CATEGORY_WEIGHTS,
        self_review=self_review or {},
        recent_48h=recent_48h,
    )

    with open(filepath, "w") as fh:
        fh.write(html)

    # Stable copy always pointing at the newest report
    latest_path = os.path.join(config.REPORT_OUTPUT_DIR, "latest.html")
    with open(latest_path, "w") as fh:
        fh.write(html)

    _prune_old_reports()

    console.print(f"\n[cyan]HTML report saved:[/cyan] {filepath}")
    return filepath


def _prune_old_reports():
    """Keep only the newest REPORT_RETENTION_COUNT timestamped reports (latest.html is exempt)."""
    reports = sorted(
        f for f in os.listdir(config.REPORT_OUTPUT_DIR)
        if f.startswith("report_") and f.endswith(".html")
    )  # filename timestamps sort chronologically
    excess = reports[:-config.REPORT_RETENTION_COUNT] if config.REPORT_RETENTION_COUNT > 0 else []
    for f in excess:
        try:
            os.remove(os.path.join(config.REPORT_OUTPUT_DIR, f))
        except OSError:
            pass
    if excess:
        console.print(f"[dim]Pruned {len(excess)} old reports (keeping newest {config.REPORT_RETENTION_COUNT})[/dim]")
