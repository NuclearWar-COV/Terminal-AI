"""
webscraper.py — Reliable universal web scraper for AI self-training.
Uses the 'requests' library for maximum compatibility.
"""

import re
import time
import json
from pathlib import Path
from collections import deque
from html.parser import HTMLParser
import urllib.parse
import urllib.robotparser
import logging

logger = logging.getLogger(__name__)

VISITED_FILE = Path(__file__).resolve().parent.parent / "scraped_urls.json"

def load_visited() -> set:
    """URLs already scraped in a previous session, so training runs can
    skip re-fetching (and re-training on) the exact same pages."""
    try:
        return set(json.loads(VISITED_FILE.read_text())) if VISITED_FILE.exists() else set()
    except Exception:
        return set()

def save_visited(urls: set):
    try:
        existing = load_visited()
        existing.update(urls)
        VISITED_FILE.write_text(json.dumps(sorted(existing), indent=2))
    except Exception:
        pass

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False


def make_session():
    if not REQUESTS_OK:
        raise ImportError("'requests' not installed. Run Setup (Run First).bat")
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    retry = Retry(total=3, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


class TextExtractor(HTMLParser):
    SKIP = {"script","style","nav","header","footer",
            "aside","noscript","form","button","iframe","svg"}
    NL   = {"p","h1","h2","h3","h4","h5","li","td",
            "blockquote","article","section","pre","br","div"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP: self._skip += 1
        if tag in self.NL:   self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP: self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if self._skip == 0:
            t = data.strip()
            if t: self.parts.append(t)

    def result(self):
        text = " ".join(self.parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_text(html):
    p = TextExtractor()
    try: p.feed(html)
    except: pass
    return p.result()

def get_title(html):
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else "Untitled"

def get_links(html, base):
    links = set()
    for href in re.findall(r'href=["\']([^"\'#][^"\']*)["\']', html, re.I):
        try:
            u = urllib.parse.urljoin(base, href).split("#")[0]
            if u.startswith("http"): links.add(u)
        except: pass
    return links


def test_connection(on_progress=None):
    if not REQUESTS_OK:
        if on_progress: on_progress("ERROR: 'requests' not installed. Run Setup (Run First).bat")
        return False
    try:
        if on_progress: on_progress("Testing internet connection...")
        r = requests.get("https://www.google.com", timeout=8, verify=False)
        if on_progress: on_progress(f"Connection OK (HTTP {r.status_code})")
        return True
    except Exception as e:
        if on_progress: on_progress(f"ERROR: No internet — {e}")
        return False


def fetch_page(session, url, timeout=15, on_error=None):
    SKIP_EXTS = (".pdf",".jpg",".jpeg",".png",".gif",".svg",".zip",
                 ".exe",".mp4",".mp3",".css",".js",".xml",".ico",".woff",".ttf")
    if any(url.lower().split("?")[0].endswith(e) for e in SKIP_EXTS):
        return None
    try:
        r = session.get(url, timeout=timeout, verify=False, allow_redirects=True)
        r.raise_for_status()
        ct = r.headers.get("Content-Type","")
        if "text" not in ct and "html" not in ct: return None
        html  = r.text
        text  = html_to_text(html)
        title = get_title(html)
        if len(text) < 80: return None
        return {"url": r.url, "title": title, "text": text, "html": html}
    except Exception as e:
        logger.warning("Could not fetch scraped page %s: %s", url, e)
        if on_error:
            on_error(f"  ERROR fetching {url}: {e}")
        return None


def scrape_url_list(urls, on_progress=None, delay=0.6, skip_visited=False):
    if not REQUESTS_OK:
        raise ImportError("'requests' not installed. Run Setup (Run First).bat")
    visited_before = load_visited() if skip_visited else set()
    session = make_session()
    results = []
    for i, url in enumerate(urls, 1):
        url = url.strip()
        if not url: continue
        if skip_visited and url in visited_before:
            if on_progress: on_progress(f"\n[{i}/{len(urls)}] {url}\n  SKIP (already scraped previously)")
            continue
        if on_progress: on_progress(f"\n[{i}/{len(urls)}] {url}")
        page = fetch_page(session, url, on_error=on_progress)
        if page:
            results.append(page)
            if on_progress: on_progress(f"  OK  {page['title'][:60]}  ({len(page['text']):,} chars)")
        else:
            if on_progress: on_progress(f"  SKIP (no content or error)")
        time.sleep(delay)
    if on_progress: on_progress(f"\nDone: {len(results)}/{len(urls)} pages scraped.")
    if skip_visited:
        save_visited({r["url"] for r in results})
    return results


def scrape_website(start_url, max_pages, on_progress=None,
                   stay_on_domain=True, delay=0.8, skip_visited=False):
    if not REQUESTS_OK:
        raise ImportError("'requests' not installed. Run Setup (Run First).bat")
    visited_before = load_visited() if skip_visited else set()
    session = make_session()
    parsed = urllib.parse.urlparse(start_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Enter a valid website URL starting with http:// or https://.")
    domain = parsed.netloc
    base   = f"{parsed.scheme}://{domain}"
    if on_progress: on_progress(f"Domain: {domain}  |  Max pages: {max_pages}")

    queue   = deque([start_url])
    visited = set()
    results = []

    try:
        sm = session.get(urllib.parse.urljoin(base, "/sitemap.xml"),
                         timeout=8, verify=False)
        locs = re.findall(r"<loc>(.*?)</loc>", sm.text, re.I)
        for u in locs[:max_pages*2]:
            u = u.strip()
            if u.startswith("http") and not u.endswith(".xml"):
                queue.append(u)
        if locs and on_progress: on_progress(f"Sitemap: {len(locs)} URLs found")
    except:
        if on_progress: on_progress("No sitemap — will follow links")

    while queue and len(results) < max_pages:
        url = queue.popleft().split("#")[0]
        if url in visited: continue
        visited.add(url)
        if stay_on_domain and urllib.parse.urlparse(url).netloc != domain:
            continue
        if skip_visited and url in visited_before:
            if on_progress: on_progress(f"\n[skip] {url[:80]} (already scraped previously)")
            continue
        if on_progress: on_progress(f"\n[{len(results)+1}/{max_pages}] {url[:80]}")
        page = fetch_page(session, url, on_error=on_progress)
        if page:
            results.append(page)
            if on_progress: on_progress(f"  OK  {page['title'][:55]}  ({len(page['text']):,} chars)")
            for lnk in get_links(page["html"], page["url"]):
                if lnk not in visited: queue.append(lnk)
        else:
            if on_progress: on_progress(f"  SKIP")
        time.sleep(delay)

    if on_progress: on_progress(f"\nCrawl done: {len(results)} pages from {domain}")
    if skip_visited:
        save_visited({r["url"] for r in results})
    return results


def save_scraped_data(results, output_dir, session_name):
    sd = Path(output_dir) / session_name
    sd.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(results):
        safe  = re.sub(r"[^\w\s-]", "", item["title"])[:50].strip() or "page"
        fname = f"{i+1:03d}_{safe}.txt"
        content = f"Source: {item['url']}\nTitle: {item['title']}\n\n{item['text']}"
        (sd / fname).write_text(content, encoding="utf-8", errors="ignore")
    meta = [{"url": r["url"], "title": r["title"], "chars": len(r["text"])} for r in results]
    (sd / "_metadata.json").write_text(json.dumps(meta, indent=2))
    return sd
