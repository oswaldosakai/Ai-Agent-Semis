"""Main agent cycle: fetch → extract → predict → recommend → persist → report."""

import uuid
from datetime import datetime

from rich.console import Console

import config
import agent.scraper as scraper
import agent.factor_analyzer as factor_analyzer
import agent.recommender as recommender
import agent.reporter as reporter
import agent.state_store as state_store

console = Console()


def run_cycle(report_html: bool = True) -> list[dict]:
    run_id = str(uuid.uuid4())[:8]
    run_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    console.print(f"\n[bold cyan]▶ Starting cycle[/bold cyan] [{run_id}] at {run_ts}")

    state_store.init_db()

    # 1. Fetch ────────────────────────────────────────────────────────────────
    console.print("  [dim]Fetching news snippets…[/dim]")
    snippets = scraper.fetch_all_news()
    console.print(f"  [dim]  → {len(snippets)} snippets collected[/dim]")

    console.print("  [dim]Fetching price data…[/dim]")
    price_data = scraper.fetch_price_data(config.ALL_SYMBOLS)

    # 2. Factor extraction ───────────────────────────────────────────────────
    console.print("  [dim]Extracting factors with Claude…[/dim]")
    try:
        current_factors = factor_analyzer.extract_factors(snippets, price_data)
        console.print(f"  [dim]  → {len(current_factors)} factors extracted[/dim]")
    except Exception as e:
        console.print(f"  [red]Factor extraction failed: {e}[/red]")
        current_factors = []

    # 3. Forward prediction ──────────────────────────────────────────────────
    console.print("  [dim]Predicting forward factors with Claude…[/dim]")
    historical_factors = state_store.get_recent_factors(n_runs=3)
    try:
        forward_factors = factor_analyzer.predict_forward_factors(current_factors, historical_factors)
        console.print(f"  [dim]  → {len(forward_factors)} forward factors predicted[/dim]")
    except Exception as e:
        console.print(f"  [red]Forward prediction failed: {e}[/red]")
        forward_factors = []

    # 4. Recommendations ─────────────────────────────────────────────────────
    console.print("  [dim]Computing recommendations…[/dim]")
    recs = recommender.compute_scores(current_factors, forward_factors, price_data)

    # 5. Persist ─────────────────────────────────────────────────────────────
    state_store.save_run(run_id, run_ts, config.CLAUDE_MODEL, len(snippets))
    state_store.save_factors(run_id, current_factors)
    state_store.save_recommendations(run_id, recs)
    state_store.purge_old_data()

    # 6. Report ──────────────────────────────────────────────────────────────
    reporter.print_cli_report(recs, current_factors, run_ts)

    if report_html:
        historical_recs = state_store.get_recent_recommendations(n_runs=10)
        reporter.generate_html_report(recs, current_factors, forward_factors, run_ts, historical_recs)

    console.print(f"[bold green]✔ Cycle {run_id} complete.[/bold green]\n")
    return recs
