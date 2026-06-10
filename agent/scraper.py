"""Web search and market data fetching layer."""

import re
import time
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Optional
import xml.etree.ElementTree as ET

import yfinance as yf

import config

try:
    from tavily import TavilyClient
    _TAVILY_AVAILABLE = True
except ImportError:
    _TAVILY_AVAILABLE = False


# ── Search queries ──────────────────────────────────────────────────────────

SEARCH_QUERIES = [
    "semiconductor industry news this week",
    "NVDA AMD INTC earnings guidance forecast",
    "US China chip tariffs export controls semiconductors",
    "AI data center GPU demand NVIDIA",
    "TSMC supply chain capacity production",
    "semiconductor inventory cycle oversupply",
    "Federal Reserve interest rates technology stocks impact",
    "geopolitical risk Taiwan chip supply",
    "ASML lithography equipment orders",
    "semiconductor capital expenditure fab investment",
    "analyst upgrades downgrades semiconductor stocks",
    "Broadcom Qualcomm Micron quarterly results",
]


# ── Tavily search ────────────────────────────────────────────────────────────

def _tavily_search(query: str, days_back: int = 14) -> list[dict]:
    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    results = client.search(
        query=query,
        search_depth="basic",
        max_results=5,
        days=days_back,
    )
    snippets = []
    for r in results.get("results", []):
        snippets.append({
            "source": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", "")[:600],
            "date": r.get("published_date", ""),
        })
    return snippets


# ── Google News RSS fallback ─────────────────────────────────────────────────

def _rss_search(query: str) -> list[dict]:
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        ns = {"dc": "http://purl.org/dc/elements/1.1/"}
        items = root.findall(".//item")
        snippets = []
        cutoff = datetime.utcnow() - timedelta(days=config.LOOKBACK_DAYS)
        for item in items[:6]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = item.findtext("pubDate") or ""
            desc = re.sub(r"<[^>]+>", "", item.findtext("description") or "")
            snippets.append({
                "source": title,
                "url": link,
                "snippet": (title + " " + desc)[:600],
                "date": pub,
            })
        return snippets
    except Exception:
        return []


# ── yfinance ticker news ─────────────────────────────────────────────────────

def _yfinance_news(tickers: list[str]) -> list[dict]:
    snippets = []
    cutoff = datetime.utcnow() - timedelta(days=config.LOOKBACK_DAYS)
    for sym in tickers[:6]:  # limit API calls
        try:
            t = yf.Ticker(sym)
            for item in (t.news or [])[:4]:
                pub_ts = item.get("providerPublishTime", 0)
                pub_dt = datetime.utcfromtimestamp(pub_ts) if pub_ts else datetime.utcnow()
                if pub_dt < cutoff:
                    continue
                content = item.get("content", {})
                title = content.get("title", item.get("title", ""))
                summary = content.get("summary", "")
                snippets.append({
                    "source": item.get("publisher", sym),
                    "url": content.get("canonicalUrl", {}).get("url", item.get("link", "")),
                    "snippet": (title + " " + summary)[:600],
                    "date": pub_dt.isoformat(),
                })
        except Exception:
            continue
    return snippets


# ── Price data ───────────────────────────────────────────────────────────────

def fetch_price_data(symbols: list[str]) -> dict:
    """Return 14-day price stats per symbol. Returns empty dicts on network failure."""
    result = {sym: {} for sym in symbols}
    end = datetime.utcnow()
    start = end - timedelta(days=config.LOOKBACK_DAYS + 5)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = yf.download(
                symbols,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        if data.empty:
            return result
        close = data["Close"] if "Close" in data.columns.get_level_values(0) else data
        for sym in symbols:
            try:
                series = (close[sym] if sym in close.columns else close).dropna()
                if len(series) < 2:
                    continue
                pct_chg = float((series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100)
                result[sym] = {
                    "price_14d_change_pct": round(pct_chg, 2),
                    "current_price": round(float(series.iloc[-1]), 2),
                    "high_14d": round(float(series.max()), 2),
                    "low_14d": round(float(series.min()), 2),
                }
            except Exception:
                pass
    except Exception:
        pass
    return result


# ── Claude knowledge fallback ────────────────────────────────────────────────

_knowledge_cache: tuple[list[dict], dict] | None = None


def _claude_knowledge_snippets() -> tuple[list[dict], dict]:
    global _knowledge_cache
    if _knowledge_cache is not None:
        return _knowledge_cache
    """Ask Claude to surface recent semiconductor market context from its knowledge."""
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    today = datetime.utcnow().date().isoformat()

    resp = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=3000,
        temperature=0.3,
        system=(
            "You are a semiconductor equity analyst with up-to-date knowledge of the market. "
            "Return ONLY valid JSON — no markdown, no explanation."
        ),
        messages=[{"role": "user", "content": f"""Today is {today}.

Generate a realistic semiconductor market intelligence report covering the last 14 days.
Include news events, earnings, macro factors, geopolitical developments, and supply/demand signals
that are plausibly current for the semiconductor sector.

Return a JSON object with two keys:

1. "snippets": array of 15-20 news items, each with:
   {{"source": string, "url": "", "snippet": string (150-300 chars), "date": "YYYY-MM-DD"}}

2. "price_estimates": object mapping each ticker to estimated 14-day price change %, e.g.:
   {{"NVDA": 8.5, "AMD": 3.2, "INTC": -2.1, "QCOM": 1.5, "AVGO": 4.0,
     "TSM": 5.5, "ASML": 2.8, "MU": 6.1, "AMAT": 3.3, "LRCX": 2.9,
     "SOXX": 4.2, "SMH": 4.5, "SOXS": -4.2, "SOXL": 8.4}}

Base estimates on realistic recent sector dynamics (AI demand cycle, China export controls,
inventory normalization, capex trends). Be specific and realistic.
"""}],
    )

    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)

    snippets = data.get("snippets", [])
    price_estimates = data.get("price_estimates", {})

    price_data = {}
    for sym in config.ALL_SYMBOLS:
        pct = price_estimates.get(sym)
        if pct is not None:
            price_data[sym] = {
                "price_14d_change_pct": round(float(pct), 2),
                "current_price": None,
                "high_14d": None,
                "low_14d": None,
                "estimated": True,
            }
        else:
            price_data[sym] = {}

    _knowledge_cache = (snippets, price_data)
    return snippets, price_data


# ── Public interface ─────────────────────────────────────────────────────────

def fetch_all_news() -> list[dict]:
    """Collect news snippets from available sources."""
    snippets: list[dict] = []
    seen_urls: set[str] = set()

    def _add(new_snippets):
        for s in new_snippets:
            url = s.get("url", "")
            if url and url in seen_urls:
                continue
            seen_urls.add(url)
            snippets.append(s)

    use_tavily = _TAVILY_AVAILABLE and bool(config.TAVILY_API_KEY)

    for query in SEARCH_QUERIES:
        if use_tavily:
            try:
                _add(_tavily_search(query, days_back=config.LOOKBACK_DAYS))
                time.sleep(0.3)
                continue
            except Exception:
                pass
        _add(_rss_search(query))
        time.sleep(0.2)

    # Supplement with yfinance per-ticker news
    _add(_yfinance_news(config.TICKERS[:6]))

    return snippets


def fetch_all_news_with_fallback() -> tuple[list[dict], bool]:
    """Fetch news; if network is blocked, fall back to Claude knowledge. Returns (snippets, used_fallback)."""
    snippets = fetch_all_news()
    if len(snippets) < 3:
        return _claude_knowledge_snippets()[0], True
    return snippets, False


def fetch_price_data_with_fallback(symbols: list[str]) -> tuple[dict, bool]:
    """Fetch prices; if network is blocked, fall back to Claude estimates. Returns (price_data, used_fallback)."""
    price_data = fetch_price_data(symbols)
    has_real = any(bool(v) for v in price_data.values())
    if not has_real:
        _, claude_prices = _claude_knowledge_snippets()
        return claude_prices, True
    return price_data, False
