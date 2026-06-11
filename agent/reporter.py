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

    template = _load_template()
    html = template.render(
        run_ts=run_ts,
        model=config.CLAUDE_MODEL,
        recommendations=recommendations,
        factors=sorted(factors, key=lambda x: abs(x.get("raw_score", 0)), reverse=True),
        forward_factors=forward_factors,
        historical_recs=historical_recs,
        buy_threshold=config.BUY_THRESHOLD,
        sell_threshold=config.SELL_THRESHOLD,
        data_estimated=data_estimated,
        accuracy_stats=accuracy_stats or {},
        weight_adjustments=weight_adjustments or {},
        category_weights=config.CATEGORY_WEIGHTS,
        self_review=self_review or {},
    )

    with open(filepath, "w") as fh:
        fh.write(html)

    console.print(f"\n[cyan]HTML report saved:[/cyan] {filepath}")
    return filepath
