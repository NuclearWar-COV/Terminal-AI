"""
search_engine.py — Multi-engine web search for Terminal AI V14.0
Supports: Google, DuckDuckGo, Bing, Yahoo
Returns list of {title, url, snippet} dicts.
"""

import re, time, json
from pathlib import Path

try:
    import requests
    import urllib3
    urllib3.disable_warnings()
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}

# ── HTML helpers ──────────────────────────────────────────────────────────────

def _strip_tags(html):
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"&amp;",  "&",  html)
    html = re.sub(r"&lt;",   "<",  html)
    html = re.sub(r"&gt;",   ">",  html)
    html = re.sub(r"&quot;", '"',  html)
    html = re.sub(r"&#39;",  "'",  html)
    html = re.sub(r"&nbsp;", " ",  html)
    html = re.sub(r"\s{2,}", " ",  html)
    return html.strip()

def _session():
    if not REQUESTS_OK:
        raise ImportError("requests not installed")
    s = requests.Session()
    s.headers.update(HEADERS)
    return s

# ── DuckDuckGo (most reliable, no API key) ────────────────────────────────────
# Uses the plain HTML endpoint (html.duckduckgo.com/html/) rather than the
# JS-driven duckduckgo.com front end. The JS version requires scraping a
# short-lived 'vqd' token out of embedded JavaScript before a second
# request even works -- that token's format has changed on DuckDuckGo's
# end multiple times and is a common source of "0 results" failures. The
# HTML endpoint returns plain, stable markup with no token dance needed.

def search_duckduckgo(query: str, n: int = 10) -> list:
    s = _session()
    r = s.post("https://html.duckduckgo.com/html/",
               data={"q": query}, timeout=10, verify=False)
    results = []
    # Each result sits in <a class="result__a" href="...">Title</a> ...
    # <a class="result__snippet" ...>snippet text</a>
    for m in re.finditer(
            r'result__a[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'result__snippet[^>]*>(.*?)</a>',
            r.text, re.DOTALL):
        url, title, snippet = m.group(1), _strip_tags(m.group(2)), _strip_tags(m.group(3))
        # DuckDuckGo's HTML endpoint wraps outbound links in a redirect
        # like //duckduckgo.com/l/?uddg=<encoded real url>&...
        if "uddg=" in url:
            import urllib.parse
            enc = re.search(r"uddg=([^&]+)", url)
            if enc:
                url = urllib.parse.unquote(enc.group(1))
        if url.startswith("http") and "duckduckgo.com" not in url:
            results.append({"url": url, "title": title or url, "snippet": snippet[:300]})
        if len(results) >= n:
            break
    return results

# ── Google (scrape, no API key) ───────────────────────────────────────────────

def search_google(query: str, n: int = 10) -> list:
    s = _session()
    r = s.get("https://www.google.com/search",
              params={"q": query, "num": n, "hl": "en"},
              timeout=10, verify=False)
    results = []
    # Extract result blocks
    for block in re.finditer(
            r'<div[^>]*class="[^"]*g[^"]*"[^>]*>(.*?)</div>\s*</div>',
            r.text, re.DOTALL):
        chunk = block.group(1)
        url_m = re.search(r'href="(https?://[^"&]+)"', chunk)
        ti_m  = re.search(r'<h3[^>]*>(.*?)</h3>', chunk, re.DOTALL)
        sn_m  = re.search(r'<span[^>]*>([\w][^<]{30,})</span>', chunk, re.DOTALL)
        if url_m and ti_m:
            url = url_m.group(1)
            if "google" not in url:
                results.append({
                    "url":     url,
                    "title":   _strip_tags(ti_m.group(1)),
                    "snippet": _strip_tags(sn_m.group(1)) if sn_m else "",
                })
        if len(results) >= n:
            break
    return results

# ── Bing (scrape) ─────────────────────────────────────────────────────────────

def search_bing(query: str, n: int = 10) -> list:
    s = _session()
    r = s.get("https://www.bing.com/search",
              params={"q": query, "count": n},
              timeout=10, verify=False)
    results = []
    for m in re.finditer(
            r'<h2><a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a></h2>',
            r.text, re.DOTALL):
        url   = m.group(1)
        title = _strip_tags(m.group(2))
        if "bing" not in url and "microsoft" not in url:
            results.append({"url": url, "title": title, "snippet": ""})
        if len(results) >= n:
            break
    return results

# ── Yahoo ─────────────────────────────────────────────────────────────────────

def search_yahoo(query: str, n: int = 10) -> list:
    s = _session()
    r = s.get("https://search.yahoo.com/search",
              params={"p": query, "n": n},
              timeout=10, verify=False)
    results = []
    for m in re.finditer(r'<h3[^>]*><a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a></h3>',
                         r.text, re.DOTALL):
        url   = m.group(1)
        title = _strip_tags(m.group(2))
        if "yahoo" not in url:
            results.append({"url": url, "title": title, "snippet": ""})
        if len(results) >= n:
            break
    return results

# ── Router ────────────────────────────────────────────────────────────────────

ENGINES = {
    "duckduckgo": search_duckduckgo,
    "google":     search_google,
    "bing":       search_bing,
    "yahoo":      search_yahoo,
}

def search(query: str, engine: str = "duckduckgo", n: int = 10, on_engine_used=None) -> list:
    """Search using the specified engine, then cascade through every other
    engine in turn if it comes back empty or errors out. Search-engine
    scraping is inherently fragile -- any one of these can break the
    moment a site changes its markup -- so trying every option before
    giving up is much more reliable than a single fallback.

    on_engine_used(name, error_or_None) is called for every engine tried,
    so callers can log exactly what happened instead of just seeing an
    unexplained empty result list."""
    tried = set()
    order = [engine.lower()] + [e for e in ENGINES if e != engine.lower()]
    for name in order:
        if name in tried or name not in ENGINES:
            continue
        tried.add(name)
        try:
            results = ENGINES[name](query, n)
            if results:
                if on_engine_used: on_engine_used(name, None)
                return results
            if on_engine_used: on_engine_used(name, "returned 0 results")
        except Exception as e:
            if on_engine_used: on_engine_used(name, str(e))
            continue
    return []

def summarize_results(query: str, results: list, page_texts: list) -> str:
    """Build a summary string from search results + fetched page text."""
    if not results and not page_texts:
        return f"No results found for '{query}'."

    parts = [f"Search results for: {query}\n"]
    for i, r in enumerate(results[:5], 1):
        parts.append(f"{i}. {r['title']}\n   {r['url']}")
        if r.get("snippet"):
            parts.append(f"   {r['snippet'][:200]}")

    if page_texts:
        parts.append("\n─── Page Summaries ───")
        for text in page_texts[:3]:
            # Take first 400 chars of each page
            snippet = text[:400].replace("\n", " ").strip()
            if snippet:
                parts.append(f"• {snippet}...")

    return "\n".join(parts)
