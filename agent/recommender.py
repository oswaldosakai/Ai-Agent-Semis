"""Scoring and Buy/Hold/Sell signal generation."""

import math
from typing import Optional
import config


def _normalize(value: float, max_val: float) -> float:
    if max_val == 0:
        return 0.0
    return max(min(value / max_val, 1.0), -1.0)


def _direction_multiplier(direction: str) -> float:
    return {
        "intensify_positive": 1.0,
        "intensify_negative": -1.0,
        "persist": 0.5,
        "diminish": 0.0,
        "reverse": -0.3,
    }.get(direction, 0.0)


def compute_scores(
    current_factors: list[dict],
    forward_factors: list[dict],
    price_data: dict,
) -> list[dict]:
    """Return a recommendation dict for each symbol."""
    symbols = config.ALL_SYMBOLS

    # ── Current scores per ticker ────────────────────────────────────────────
    current_totals: dict[str, float] = {s: 0.0 for s in symbols}
    current_counts: dict[str, int] = {s: 0 for s in symbols}

    for f in current_factors:
        score = f.get("raw_score", 0.0)
        for ticker in f.get("affected_tickers", []):
            if ticker in current_totals:
                current_totals[ticker] += score
                current_counts[ticker] += 1

    # ── Forward scores per ticker ────────────────────────────────────────────
    forward_totals: dict[str, float] = {s: 0.0 for s in symbols}

    for ff in forward_factors:
        direction = ff.get("expected_direction", "persist")
        confidence = float(ff.get("confidence", 0.5))
        magnitude = float(ff.get("time_horizon_days", 7)) / 14.0
        dmul = _direction_multiplier(direction)
        raw_fwd = dmul * confidence * magnitude * 5  # scale to ~factor score range
        w = config.CATEGORY_WEIGHTS.get(ff.get("category", ""), 1.0)
        for ticker in ff.get("affected_tickers", []):
            if ticker in forward_totals:
                forward_totals[ticker] += raw_fwd * w

    # ── Max for normalization ────────────────────────────────────────────────
    all_current = list(current_totals.values())
    all_forward = list(forward_totals.values())
    max_current = max((abs(v) for v in all_current), default=1.0) or 1.0
    max_forward = max((abs(v) for v in all_forward), default=1.0) or 1.0

    recommendations = []
    for sym in symbols:
        norm_current = _normalize(current_totals[sym], max_current)
        norm_forward = _normalize(forward_totals[sym], max_forward)
        composite = 0.6 * norm_current + 0.4 * norm_forward

        # Price momentum nudge (±0.1)
        pdata = price_data.get(sym, {})
        pct_chg = pdata.get("price_14d_change_pct", 0.0) or 0.0
        momentum = max(min(pct_chg / 100.0, 0.1), -0.1)
        composite = max(min(composite + momentum, 1.0), -1.0)

        # Flip signal for inverse ETFs
        display_composite = composite
        if sym in config.INVERSE_ETFS:
            display_composite = -composite

        # Signal mapping
        if display_composite >= config.BUY_THRESHOLD:
            signal = "BUY"
        elif display_composite <= config.SELL_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Confidence = distance from nearest threshold
        dist_buy = abs(display_composite - config.BUY_THRESHOLD)
        dist_sell = abs(display_composite - config.SELL_THRESHOLD)
        confidence = 1.0 - min(dist_buy, dist_sell) / max(config.BUY_THRESHOLD, 0.01)
        confidence = round(max(min(confidence, 1.0), 0.0), 2)

        # Top factors for this ticker
        top_factors = sorted(
            [f for f in current_factors if sym in f.get("affected_tickers", [])],
            key=lambda x: abs(x.get("raw_score", 0)),
            reverse=True,
        )[:3]

        recommendations.append({
            "ticker": sym,
            "current_score": round(norm_current, 3),
            "forward_score": round(norm_forward, 3),
            "composite_score": round(display_composite, 3),
            "signal": signal,
            "confidence": confidence,
            "top_factors": [
            {"description": f.get("description", ""), "event_date": f.get("event_date", "")}
            for f in top_factors
        ],
            "price_14d_change": pct_chg,
        })

    return recommendations
