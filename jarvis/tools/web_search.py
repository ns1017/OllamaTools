"""Web search tool: DuckDuckGo search with smart news/weather routing, plus a
short-lived cache so the model re-asking the same question doesn't fire a
duplicate search."""
import random
import time

from ddgs import DDGS

from jarvis.config import CONFIG
from jarvis.logging_utils import log_event

_last_search = {"query": None, "mode": None, "time": 0, "results": ""}


def get_last_search() -> dict:
    """Read-only accessor for the last-search cache (used for logging and
    repeat-detection fallbacks in the tool loop)."""
    return dict(_last_search)


def perform_web_search(query: str, max_results: int = None, mode: str = None) -> str:
    """Perform web search using DuckDuckGo with smart routing for news and weather.

    `mode` ("news", "weather", or "general") lets a caller that already knows
    what kind of search it wants skip the keyword-based guessing below
    entirely — useful when the query text itself doesn't contain a clean
    signal word, or contains a misleading one."""
    max_results = max_results if max_results is not None else CONFIG.get("web_search", "max_results", default=3)

    if mode in ("news", "weather", "general"):
        is_news = mode == "news"
        is_weather = mode == "weather"
    else:
        is_news = any(kw in query.lower() for kw in ["news", "latest", "current events", "breaking", "headlines", "what's happening", "what is happening", "what are the latest", "what's new", "what is new", "what are the news", "what's the news", "what is the news", ])
        is_weather = any(kw in query.lower() for kw in ["weather", "temperature", "forecast", "forecasts", ])

    mode = "news" if is_news else "weather" if is_weather else "general"
    log_event("web_search:start", {"query": query, "max_results": max_results, "mode": mode})

    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                if is_news:
                    print(f"📰 Using news search for: {query}")
                    results = list(ddgs.news(query, max_results=max_results))
                    output = [f" Latest news results for: '{query}'\n"]
                    for i, r in enumerate(results, 1):
                        date_str = r.get("date", "N/A")
                        output.append(
                            f"{i}. {r.get('title', 'No title')} ({date_str})\n"
                            f"   {r.get('body', '')[:280]}...\n"
                            f"   Source: {r.get('source', r.get('href', 'Unknown'))}\n"
                        )
                elif is_weather:
                    refined = f"{query} (weather.com OR accuweather.com OR bbc.com/weather OR noaa.gov)"
                    print(f"🌤️ Using targeted weather search for: {query}")
                    results = list(ddgs.text(refined, max_results=max_results))
                    output = [f" Weather results for: '{query}'\n"]
                    for i, r in enumerate(results, 1):
                        output.append(
                            f"{i}. {r.get('title', 'No title')}\n"
                            f"   {r.get('body', '')[:280]}...\n"
                            f"   Source: {r.get('href', 'Unknown')}\n"
                        )
                else:
                    print(f"🔍 General web search for: {query}")
                    results = list(ddgs.text(query, max_results=max_results))
                    output = [f"Web search results for: '{query}'\n"]
                    for i, r in enumerate(results, 1):
                        output.append(
                            f"{i}. {r.get('title', 'No title')}\n"
                            f"   {r.get('body', '')[:250]}...\n"
                            f"   Source: {r.get('href', 'Unknown')}\n"
                        )

                return "\n".join(output)

        except Exception as e:
            print(f"Search attempt {attempt+1}/3 failed: {e}")
            log_event("web_search:attempt_error", {"attempt": attempt + 1, "error": str(e)})
            if attempt < 2:
                sleep_time = 1.2 * (attempt + 1) + random.uniform(0.3, 0.8)
                time.sleep(sleep_time)
            else:
                return f"Search error after retries: {str(e)[:200]}"

    return "Search failed completely."


def search_with_cache(query: str, ttl_seconds: int = None, mode: str = None):
    """Return (results, was_cache_hit) for `query`, reusing the last result if
    it's the same query (and mode) within ttl_seconds (default from
    config.json -> web_search.cache_ttl_seconds)."""
    ttl_seconds = ttl_seconds if ttl_seconds is not None else CONFIG.get("web_search", "cache_ttl_seconds", default=30)
    now = time.time()
    if (query == _last_search.get("query") and mode == _last_search.get("mode")
            and now - _last_search.get("time", 0) < ttl_seconds):
        log_event("web_search:cache_hit", {"query": query, "mode": mode})
        return _last_search.get("results", ""), True

    results = perform_web_search(query, mode=mode)
    _last_search["query"] = query
    _last_search["mode"] = mode
    _last_search["time"] = now
    _last_search["results"] = results
    log_event("web_search:results", results)
    return results, False
