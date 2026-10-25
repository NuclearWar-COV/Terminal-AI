"""
search_engine.py — Multi-engine web search for Terminal AI V14.0
Supports: Google, DuckDuckGo, Bing, Yahoo
Returns list of {title, url, snippet} dicts.
"""

import re
import urllib.parse
from html.parser import HTMLParser

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

class _ResultLinkParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.seen = set()
        self.heading_depth = 0
        self.anchor = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        if tag in ("h2", "h3"):
            self.heading_depth += 1
            if self.anchor is not None:
                self.anchor["candidate"] = True
        if tag == "a" and attrs.get("href"):
            self.anchor = {"url": attrs["href"], "parts": [],
                           "candidate": bool(classes & {"result__a"}) or self.heading_depth > 0}

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data):
        if self.anchor is not None:
            self.anchor["parts"].append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.anchor is not None:
            item = self.anchor
            self.anchor = None
            url = item["url"]
            raw_url = urllib.parse.urlparse(url)
            if raw_url.path == "/url":
                params = urllib.parse.parse_qs(raw_url.query)
                url = params.get("q", params.get("url", [url]))[0]
            if "uddg=" in url:
                query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                url = query.get("uddg", [url])[0]
            url = urllib.parse.unquote(url)
            parsed = urllib.parse.urlparse(url)
            title = " ".join("".join(item["parts"]).split())
            if (item["candidate"] and parsed.scheme in ("http", "https") and parsed.netloc
                    and not any(host in parsed.netloc.casefold() for host in
                                ("duckduckgo.com", "google.", "bing.com", "yahoo.com", "microsoft.com"))
                    and url not in self.seen and title):
                self.seen.add(url)
                self.results.append({"url": url, "title": title, "snippet": ""})
        elif tag in ("h2", "h3"):
            self.heading_depth = max(0, self.heading_depth - 1)

def _parse_result_links(html, limit):
    parser = _ResultLinkParser()
    parser.feed(html)
    return parser.results[:limit]

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
    r.raise_for_status()
    return _parse_result_links(r.text, n)

# ── Google (scrape, no API key) ───────────────────────────────────────────────

def search_google(query: str, n: int = 10) -> list:
    s = _session()
    r = s.get("https://www.google.com/search",
              params={"q": query, "num": n, "hl": "en"},
              timeout=10, verify=False)
    r.raise_for_status()
    return _parse_result_links(r.text, n)

# ── Bing (scrape) ─────────────────────────────────────────────────────────────

def search_bing(query: str, n: int = 10) -> list:
    s = _session()
    r = s.get("https://www.bing.com/search",
              params={"q": query, "count": n},
              timeout=10, verify=False)
    r.raise_for_status()
    return _parse_result_links(r.text, n)

# ── Yahoo ─────────────────────────────────────────────────────────────────────

def search_yahoo(query: str, n: int = 10) -> list:
    s = _session()
    r = s.get("https://search.yahoo.com/search",
              params={"p": query, "n": n},
              timeout=10, verify=False)
    r.raise_for_status()
    return _parse_result_links(r.text, n)

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
