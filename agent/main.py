#!/usr/bin/env python3
"""Entry point — CLI flags and scheduler."""

import sys
import os
import argparse
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich import box
from apscheduler.schedulers.blocking import BlockingScheduler

import config
import agent.state_store as state_store
import agent.orchestrator as orchestrator

console = Console()


def _open_report(report_path):
    """Open the HTML report in the default browser (skipped in CI/headless runs)."""
    if not report_path:
        return
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return  # no browser on GitHub Actions runners
    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(report_path)}")
    console.print("[cyan]Report opened in your browser.[/cyan]")


def cmd_run_now(args):
    _, report_path = orchestrator.run_cycle(report_html=not args.no_html)
    _open_report(report_path)


def cmd_schedule(args):
    state_store.init_db()
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        lambda: orchestrator.run_cycle(report_html=True),
        trigger="interval",
        hours=config.SCHEDULE_HOURS,
        id="semis_agent",
        next_run_time=datetime.utcnow(),
    )
    console.print(
        f"[bold cyan]Scheduler started[/bold cyan] — "
        f"running every [yellow]{config.SCHEDULE_HOURS}h[/yellow]. "
        "Press Ctrl+C to stop."
    )
    try:
        scheduler.start()
    except KeyboardInterrupt:
        console.print("\n[yellow]Scheduler stopped.[/yellow]")


def cmd_history(args):
    state_store.init_db()
    recs = state_store.get_recent_recommendations(n_runs=int(args.runs))
    if not recs:
        console.print("[yellow]No history found.[/yellow]")
        return

    table = Table(title=f"Last {args.runs} cycles — signal history", box=box.SIMPLE)
    table.add_column("Timestamp", style="dim")
    table.add_column("Ticker")
    table.add_column("Signal")
    table.add_column("Score")

    for r in recs:
        sig = r.get("signal", "?")
        color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(sig, "white")
        table.add_row(
            r.get("timestamp", "")[:16],
            r.get("ticker", ""),
            f"[{color}]{sig}[/{color}]",
            f"{r.get('composite_score', 0):+.3f}",
        )
    console.print(table)


def main():
    state_store.init_db()
    parser = argparse.ArgumentParser(
        description="Semiconductor Factor Analysis Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent/main.py --run-now           # Single analysis run
  python agent/main.py --schedule          # Run every SCHEDULE_HOURS hours
  python agent/main.py --history --runs 5  # Show last 5 cycle signals
  python agent/main.py --run-now --no-html # CLI output only
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    run_p = subparsers.add_parser("--run-now", help="Run a single cycle now")
    run_p.add_argument("--no-html", action="store_true", help="Skip HTML report")

    sched_p = subparsers.add_parser("--schedule", help="Run on a recurring schedule")

    hist_p = subparsers.add_parser("--history", help="Show signal history")
    hist_p.add_argument("--runs", default="10", help="Number of past runs to show")

    # Support flat flags too (argparse subcommands require exact match)
    args, _ = parser.parse_known_args()

    if "--schedule" in sys.argv:
        cmd_schedule(argparse.Namespace())
    elif "--history" in sys.argv:
        runs = "10"
        if "--runs" in sys.argv:
            idx = sys.argv.index("--runs")
            if idx + 1 < len(sys.argv):
                runs = sys.argv[idx + 1]
        cmd_history(argparse.Namespace(runs=runs))
    elif "--run-now" in sys.argv or len(sys.argv) == 1:
        no_html = "--no-html" in sys.argv
        cmd_run_now(argparse.Namespace(no_html=no_html))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
