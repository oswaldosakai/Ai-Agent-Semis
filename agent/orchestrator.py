"""Main agent cycle: tune → fetch → backtest → extract → research → predict → recommend → persist → report → self-review."""

import uuid
from datetime import datetime

from rich.console import Console

import config
import agent.scraper as scraper
import agent.factor_analyzer as factor_analyzer
import agent.recommender as recommender
import agent.reporter as reporter
import agent.state_store as state_store
import agent.backtester as backtester
import agent.researcher as researcher
import agent.self_review as self_review

console = Console()


def run_cycle(report_html: bool = True) -> list[dict]:
    run_id = str(uuid.uuid4())[:8]
    run_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    errors: list[str] = []

    console.print(f"\n[bold cyan]▶ Starting cycle[/bold cyan] [{run_id}] at {run_ts}")

    # 0. Apply tuning learned from previous self-reviews ──────────────────────
    persisted = self_review.load_persisted_tuning()
    if persisted:
        console.print(f"  [dim]Loaded persisted tuning from previous self-reviews: "
                      f"buy≥{config.BUY_THRESHOLD} sell≤{config.SELL_THRESHOLD}[/dim]")

    # 1. Fetch ────────────────────────────────────────────────────────────────
    console.print("  [dim]Fetching news snippets…[/dim]")
    snippets, news_fallback = scraper.fetch_all_news_with_fallback()
    if news_fallback:
        console.print(f"  [yellow]  → Network unavailable — using Claude knowledge base ({len(snippets)} items)[/yellow]")
    else:
        console.print(f"  [dim]  → {len(snippets)} snippets collected[/dim]")

    console.print("  [dim]Fetching price data…[/dim]")
    if news_fallback:
        try:
            _, price_data = scraper._claude_knowledge_snippets()
            console.print("  [yellow]  → Price data estimated by Claude[/yellow]")
        except Exception as e:
            errors.append(f"price fallback failed: {e}")
            price_data = {sym: {} for sym in config.ALL_SYMBOLS}
    else:
        price_data, price_fallback = scraper.fetch_price_data_with_fallback(config.ALL_SYMBOLS)
        if price_fallback:
            console.print("  [yellow]  → Price data estimated by Claude[/yellow]")

    # 2. Backtest previous signals ────────────────────────────────────────────
    state_store.init_db()
    accuracy_stats, weight_adjustments = backtester.run_backtest(price_data)
    if accuracy_stats:
        backtester.print_accuracy_table(accuracy_stats)

    # 3. Factor extraction ───────────────────────────────────────────────────
    console.print("  [dim]Extracting factors with Claude…[/dim]")
    try:
        current_factors = factor_analyzer.extract_factors(snippets, price_data)
        console.print(f"  [dim]  → {len(current_factors)} factors extracted[/dim]")
    except Exception as e:
        console.print(f"  [red]Factor extraction failed: {e}[/red]")
        errors.append(f"factor extraction failed: {e}")
        current_factors = []

    # 4. Deep-dive narratives for top 3 factors ──────────────────────────────
    if current_factors:
        console.print("  [dim]Researching top 3 factors in depth…[/dim]")
        try:
            current_factors = researcher.research_top_factors(current_factors, top_n=3)
        except Exception as e:
            console.print(f"  [yellow]  → Factor research skipped: {e}[/yellow]")
            errors.append(f"factor research skipped: {e}")

    # 5. Forward prediction ──────────────────────────────────────────────────
    console.print("  [dim]Predicting forward factors with Claude…[/dim]")
    historical_factors = state_store.get_recent_factors(n_runs=3)
    try:
        forward_factors = factor_analyzer.predict_forward_factors(current_factors, historical_factors)
        console.print(f"  [dim]  → {len(forward_factors)} forward factors predicted[/dim]")
    except Exception as e:
        console.print(f"  [red]Forward prediction failed: {e}[/red]")
        errors.append(f"forward prediction failed: {e}")
        forward_factors = []

    # 6. Recommendations ─────────────────────────────────────────────────────
    console.print("  [dim]Computing recommendations…[/dim]")
    recs = recommender.compute_scores(current_factors, forward_factors, price_data)

    # 7. Persist ─────────────────────────────────────────────────────────────
    state_store.save_run(run_id, run_ts, config.CLAUDE_MODEL, len(snippets))
    state_store.save_factors(run_id, current_factors)
    state_store.save_recommendations(run_id, recs)
    state_store.purge_old_data()

    # 8. Self-review: audit this run's outputs and tune for the next ─────────
    review = {}
    try:
        summary = self_review.build_cycle_summary(
            snippets_count=len(snippets),
            used_fallback=news_fallback,
            factors=current_factors,
            forward_factors=forward_factors,
            recommendations=recs,
            accuracy_stats=accuracy_stats,
            errors=errors,
        )
        review = self_review.run_self_review(summary)
    except Exception as e:
        console.print(f"  [yellow]Self-review skipped: {e}[/yellow]")

    # 9. Report ──────────────────────────────────────────────────────────────
    reporter.print_cli_report(recs, current_factors, run_ts,
                              data_note="[Estimated via Claude knowledge — live data unavailable]" if news_fallback else None)

    if report_html:
        historical_recs = state_store.get_recent_recommendations(n_runs=10)
        reporter.generate_html_report(
            recs, current_factors, forward_factors, run_ts,
            historical_recs, data_estimated=news_fallback,
            accuracy_stats=accuracy_stats, weight_adjustments=weight_adjustments,
            self_review=review,
        )

    console.print(f"[bold green]✔ Cycle {run_id} complete.[/bold green]\n")
    return recs
