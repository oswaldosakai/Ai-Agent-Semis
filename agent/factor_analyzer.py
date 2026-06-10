"""Claude-powered factor extraction and scoring."""

import json
import math
import re
from datetime import datetime
from typing import Optional

import anthropic
import config

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


EXTRACTION_SYSTEM = """You are a semiconductor equity analyst. Your task is to extract
structured investment factors from news snippets and price data.

Rules:
- Identify only concrete, evidence-backed factors (not speculation)
- Merge duplicate events from multiple snippets into one factor
- Be conservative: magnitude 5 is reserved for major earnings misses/beats or war-level geopolitical events
- Output ONLY valid JSON matching the schema — no markdown, no explanation
"""

EXTRACTION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["category", "description", "sentiment", "magnitude", "recency_days", "event_date", "affected_tickers", "evidence"],
        "properties": {
            "category": {"type": "string", "enum": config.FACTOR_CATEGORIES},
            "description": {"type": "string"},
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "magnitude": {"type": "integer", "minimum": 1, "maximum": 5},
            "recency_days": {"type": "number"},
            "event_date": {"type": "string", "description": "Date the event occurred, YYYY-MM-DD"},
            "affected_tickers": {"type": "array", "items": {"type": "string"}},
            "evidence": {"type": "array", "items": {"type": "string"}},
        },
    },
}

PREDICTION_SYSTEM = """You are a semiconductor sector strategist specializing in factor
persistence and mean-reversion dynamics.

Given a list of current factors and their historical trajectory, predict which factors
will intensify or diminish over the next 1–2 weeks.

Output ONLY valid JSON matching the schema — no markdown, no explanation.
"""

PREDICTION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["category", "description", "expected_direction", "confidence", "time_horizon_days", "affected_tickers", "reasoning"],
        "properties": {
            "category": {"type": "string"},
            "description": {"type": "string"},
            "expected_direction": {"type": "string", "enum": ["intensify_positive", "intensify_negative", "diminish", "persist", "reverse"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "time_horizon_days": {"type": "integer"},
            "affected_tickers": {"type": "array", "items": {"type": "string"}},
            "reasoning": {"type": "string"},
        },
    },
}


def _call_claude(system: str, user: str, temperature: float = 0.2) -> str:
    client = _get_client()
    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=4096,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def _parse_json_safe(text: str) -> list:
    """Extract JSON array from Claude response, tolerating markdown fences."""
    text = text.strip()
    # strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _recency_decay(days: float) -> float:
    return math.exp(-0.1 * max(0, days))


def extract_factors(snippets: list[dict], price_data: dict) -> list[dict]:
    """Call Claude to extract factors from news snippets + price data."""
    today = datetime.utcnow().date().isoformat()

    price_summary = "\n".join(
        f"  {sym}: {d.get('price_14d_change_pct', 'N/A')}% over 14d, current ${d.get('current_price', 'N/A')}"
        for sym, d in price_data.items() if d
    )

    snippets_text = "\n\n".join(
        f"[{i+1}] Source: {s.get('source','?')} | Date: {s.get('date','?')}\n{s.get('snippet','')}"
        for i, s in enumerate(snippets[:60])
    )

    user_prompt = f"""Today's date: {today}
Universe: {', '.join(config.ALL_SYMBOLS)}
Factor categories: {', '.join(config.FACTOR_CATEGORIES)}

=== PRICE DATA (last 14 days) ===
{price_summary or 'Unavailable'}

=== NEWS SNIPPETS ===
{snippets_text}

=== INSTRUCTIONS ===
Extract all significant investment factors from the snippets above.
- Set recency_days to the approximate age of the event in days (0=today, 7=one week ago)
- Set event_date to the actual calendar date the event occurred (YYYY-MM-DD format)
- affected_tickers should list only symbols from the universe above
- If a factor affects the whole sector, list all relevant tickers
- evidence should be 1-3 short quotes from the snippets

Return a JSON array following this schema:
{json.dumps(EXTRACTION_SCHEMA, indent=2)}
"""

    raw = _call_claude(EXTRACTION_SYSTEM, user_prompt)
    factors = _parse_json_safe(raw)

    sentiment_val = {"positive": 1, "negative": -1, "neutral": 0}
    for f in factors:
        w = config.CATEGORY_WEIGHTS.get(f.get("category", ""), 1.0)
        s = sentiment_val.get(f.get("sentiment", "neutral"), 0)
        m = f.get("magnitude", 1)
        d = f.get("recency_days", 0)
        f["raw_score"] = s * m * _recency_decay(d) * w

    return factors


def predict_forward_factors(current_factors: list[dict], historical_factors: list[dict]) -> list[dict]:
    """Call Claude to predict how factors will evolve over the next 1–2 weeks."""
    current_text = json.dumps(
        [{"category": f.get("category"), "description": f.get("description"),
          "sentiment": f.get("sentiment"), "magnitude": f.get("magnitude"),
          "affected_tickers": f.get("affected_tickers", [])}
         for f in current_factors],
        indent=2,
    )

    # Summarize historical factor categories for context
    hist_summary = {}
    for f in historical_factors:
        key = f.get("category", "")
        hist_summary.setdefault(key, {"count": 0, "sentiments": []})
        hist_summary[key]["count"] += 1
        hist_summary[key]["sentiments"].append(f.get("sentiment", "neutral"))

    hist_text = json.dumps(hist_summary, indent=2)

    user_prompt = f"""=== CURRENT FACTORS (this cycle) ===
{current_text}

=== HISTORICAL FACTOR PATTERNS (last 3 cycles) ===
{hist_text}

Universe: {', '.join(config.ALL_SYMBOLS)}

Analyze which factors will intensify, persist, or diminish over the next 1–14 days.
Focus on: event timelines (upcoming earnings, scheduled Fed meetings, supply deadlines),
trend momentum, and mean-reversion dynamics.

Return a JSON array following this schema:
{json.dumps(PREDICTION_SCHEMA, indent=2)}
"""

    raw = _call_claude(PREDICTION_SYSTEM, user_prompt)
    forward_factors = _parse_json_safe(raw)
    return forward_factors
