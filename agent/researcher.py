"""
Deep-dive researcher: takes the top N factors from a cycle, searches for more
context, and asks Claude to write a 1–2 paragraph narrative for each.
"""

import json
import urllib.parse
import urllib.request
import re
import time
from typing import Optional

import anthropic
from rich.console import Console

import config

console = Console()

_tavily_available = False
try:
    from tavily import TavilyClient
    _tavily_available = True
except ImportError:
    pass


# ── Web search helpers ────────────────────────────────────────────────────────

def _tavily_search(query: str) -> list[str]:
    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    results = client.search(query=query, search_depth="advanced", max_results=4)
    return [r.get("content", "")[:800] for r in results.get("results", [])]


def _rss_search(query: str) -> list[str]:
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(raw)
        snippets = []
        for item in root.findall(".//item")[:5]:
            title = item.findtext("title") or ""
            desc = re.sub(r"<[^>]+>", "", item.findtext("description") or "")
            snippets.append((title + " " + desc)[:600])
        return snippets
    except Exception:
        return []


def _search(query: str) -> list[str]:
    use_tavily = _tavily_available and bool(config.TAVILY_API_KEY)
    if use_tavily:
        try:
            return _tavily_search(query)
        except Exception:
            pass
    return _rss_search(query)


# ── Factor query builder ──────────────────────────────────────────────────────

def _build_queries(factor: dict) -> list[str]:
    desc = factor.get("description", "")
    category = factor.get("category", "")
    tickers = factor.get("affected_tickers", [])[:3]
    ticker_str = " ".join(tickers)

    # Strip leading date prefix like "[2026-06-05] " for cleaner queries
    clean_desc = re.sub(r"^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s*", "", desc)
    short_desc = clean_desc[:80]

    queries = [f"{short_desc} semiconductor 2026"]
    if ticker_str:
        queries.append(f"{ticker_str} {category} impact June 2026")
    if category in ("earnings", "analyst_actions"):
        queries.append(f"{ticker_str} earnings guidance analyst 2026")
    elif category == "tariffs":
        queries.append("US chip export controls tariffs semiconductor 2026")
    elif category == "geopolitics":
        queries.append("geopolitical risk semiconductor supply chain 2026")
    elif category == "ai_demand":
        queries.append("AI data center GPU demand hyperscaler capex 2026")
    elif category in ("supply_chain", "inventory"):
        queries.append("semiconductor supply chain inventory cycle 2026")
    elif category == "rates":
        queries.append("Federal Reserve interest rates semiconductor stocks 2026")

    return queries[:2]  # limit to 2 searches per factor to manage API cost


# ── Narrative writer ──────────────────────────────────────────────────────────

NARRATIVE_SYSTEM = """You are a senior semiconductor equity analyst writing a research note.
Your writing is precise, evidence-based, and directly useful to an investor.
Avoid generic statements. Cite specific companies, dates, and figures where available.
Do not use bullet points — write in flowing prose.
"""


def _write_narrative(factor: dict, search_results: list[str], today: str) -> str:
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    search_text = "\n\n---\n\n".join(search_results) if search_results else "No additional search results available."
    tickers = ", ".join(factor.get("affected_tickers", []))
    evidence = "\n".join(f'- "{e}"' for e in (factor.get("evidence") or [])[:3])

    prompt = f"""Today: {today}

Factor to analyze:
  Category: {factor.get("category")}
  Description: {factor.get("description")}
  Sentiment: {factor.get("sentiment")} (magnitude {factor.get("magnitude")}/5)
  Event date: {factor.get("event_date", "recent")}
  Affected tickers: {tickers}

Original evidence snippets:
{evidence or "None"}

Additional research context:
{search_text}

Write 1–2 dense, analyst-quality paragraphs (150–250 words total) that:
1. Explain what happened and why it matters for semiconductor stocks
2. Identify which companies are most exposed and in what direction
3. Describe what to watch for in the next 1–2 weeks that would confirm or contradict this factor

Write directly — no title, no headers, no bullet points.
"""

    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=512,
        temperature=0.3,
        system=NARRATIVE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


# ── Public interface ──────────────────────────────────────────────────────────

def research_top_factors(factors: list[dict], top_n: int = 3) -> list[dict]:
    """
    Take the top_n factors by abs(raw_score), deep-search each, write a narrative.
    Returns the same factor dicts with a 'narrative' key added.
    """
    from datetime import datetime
    today = datetime.utcnow().date().isoformat()

    top = sorted(factors, key=lambda x: abs(x.get("raw_score", 0)), reverse=True)[:top_n]
    results = []

    for i, factor in enumerate(top):
        desc_short = factor.get("description", "")[:60]
        console.print(f"  [dim]  → Researching factor {i+1}/{top_n}: {desc_short}…[/dim]")

        queries = _build_queries(factor)
        search_results = []
        for q in queries:
            hits = _search(q)
            search_results.extend(hits)
            time.sleep(0.2)

        try:
            narrative = _write_narrative(factor, search_results, today)
        except Exception as e:
            narrative = f"[Narrative unavailable: {e}]"

        enriched = dict(factor)
        enriched["narrative"] = narrative
        enriched["search_hit_count"] = len(search_results)
        results.append(enriched)

    # Attach narratives back onto the original factor dicts by description key
    narrative_by_desc = {r.get("description", ""): r.get("narrative", "") for r in results}
    for f in factors:
        narrative = narrative_by_desc.get(f.get("description", ""))
        if narrative:
            f["narrative"] = narrative

    return factors
