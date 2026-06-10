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


def print_cli_report(recommendations: list[dict], factors: list[dict], run_ts: str):
    console.rule(f"[bold cyan]Semis Factor Analysis — {run_ts}[/bold cyan]")

    # ── Signals table ────────────────────────────────────────────────────────
    table = Table(title="Signals", box=box.ROUNDED, show_lines=True)
    table.add_column("Ticker", style="bold white", width=8)
    table.add_column("Signal", width=6)
    table.add_column("Score", width=7)
    table.add_column("14d %", width=8)
    table.add_column("Top Factors", min_width=40)

    for r in sorted(recommendations, key=lambda x: x["composite_score"], reverse=True):
        sig = r["signal"]
        table.add_row(
            r["ticker"],
            Text(sig, style=_signal_style(sig)),
            f"{r['composite_score']:+.3f}",
            f"{r.get('price_14d_change', 0):+.1f}%",
            "; ".join(r.get("top_factors", [])[:2]),
        )
    console.print(table)

    # ── Top factors ──────────────────────────────────────────────────────────
    console.rule("[bold]Key Factors This Cycle[/bold]")
    top = sorted(factors, key=lambda x: abs(x.get("raw_score", 0)), reverse=True)[:8]
    for f in top:
        sentiment_icon = {"positive": "↑", "negative": "↓", "neutral": "→"}.get(f.get("sentiment"), "")
        color = {"positive": "green", "negative": "red", "neutral": "white"}.get(f.get("sentiment"), "white")
        console.print(
            f"  [{color}]{sentiment_icon} [{f.get('category','?')}][/{color}] "
            f"{f.get('description','')[:120]} "
            f"[dim](mag={f.get('magnitude')}, {f.get('recency_days',0):.0f}d ago)[/dim]"
        )


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
    )

    with open(filepath, "w") as fh:
        fh.write(html)

    console.print(f"\n[cyan]HTML report saved:[/cyan] {filepath}")
    return filepath
