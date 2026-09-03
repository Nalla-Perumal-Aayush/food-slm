"""Self-hosted fetch (Documents/mcp_extraction_pipeline.md §1 layer 3, §5).

No scraping SaaS. One URL at a time — never a crawl. `robots.txt` is enforced,
requests are rate-limited per domain, and every response is cached by URL-hash so
a page is fetched at most once. `trafilatura` pulls the main content out of the
HTML.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.robotparser
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_UA = "food-slm-research/0.1 (culinary knowledge extraction; non-commercial)"


@dataclass
class FetchResult:
    url: str
    status: int                    # HTTP status, or -1 blocked-by-robots, -2 fetch error
    html: str = ""
    text: str = ""                 # trafilatura main-content extraction
    from_cache: bool = False
    fetched_at: str = ""
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200 and bool(self.text)


class RobotsCache:
    def __init__(self, user_agent: str = DEFAULT_UA) -> None:
        self.ua = user_agent
        self._parsers: dict[str, urllib.robotparser.RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        import httpx

        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        rp = self._parsers.get(origin)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = httpx.get(f"{origin}/robots.txt", timeout=15,
                              headers={"User-Agent": self.ua}, follow_redirects=True)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except Exception:  # noqa: BLE001 - no robots.txt => allow
                rp.parse([])
            self._parsers[origin] = rp
        return rp.can_fetch(self.ua, url)


@dataclass
class RateLimiter:
    min_interval_s: float = 2.0
    _last: dict[str, float] = field(default_factory=dict)

    def wait(self, url: str) -> None:
        host = urlparse(url).netloc
        elapsed = time.monotonic() - self._last.get(host, 0.0)
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last[host] = time.monotonic()


class Fetcher:
    """Robots-respecting, rate-limited, cached single-page fetcher."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        user_agent: str = DEFAULT_UA,
        min_interval_s: float = 2.0,
        timeout_s: float = 30.0,
        obey_robots: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ua = user_agent
        self.timeout = timeout_s
        self.limiter = RateLimiter(min_interval_s)
        self.robots = RobotsCache(user_agent) if obey_robots else None

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.json"

    def fetch(self, url: str, *, refresh: bool = False) -> FetchResult:
        cp = self._cache_path(url)
        if cp.exists() and not refresh:
            d = json.loads(cp.read_text(encoding="utf-8"))
            return FetchResult(**{**d, "from_cache": True})

        if self.robots is not None and not self.robots.allowed(url):
            return FetchResult(url=url, status=-1, reason="disallowed by robots.txt",
                               fetched_at=datetime.now(UTC).isoformat(timespec="seconds"))

        import httpx

        self.limiter.wait(url)
        try:
            r = httpx.get(url, headers={"User-Agent": self.ua}, timeout=self.timeout,
                          follow_redirects=True)
        except Exception as exc:  # noqa: BLE001
            return FetchResult(url=url, status=-2, reason=f"{type(exc).__name__}: {exc}",
                               fetched_at=datetime.now(UTC).isoformat(timespec="seconds"))

        text = ""
        if r.status_code == 200:
            import trafilatura

            text = trafilatura.extract(r.text, include_comments=False, include_tables=True,
                                       favor_precision=True) or ""

        res = FetchResult(
            url=url, status=r.status_code, html=r.text if r.status_code == 200 else "",
            text=text, fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        cp.write_text(json.dumps(res.__dict__, ensure_ascii=False), encoding="utf-8")
        return res
