"""Keyless video B-roll fetcher for philosopher reels.

Pulls short, license-clean motion footage themed to a philosopher / quote
from two keyless sources:

  1. Wikimedia Commons  (File namespace, video MIME only)  -- primary
  2. Archive.org        (advancedsearch + metadata, movies) -- fallback

This is a clean-room module written for this repo. It deliberately mirrors
the polite-fetch conventions already used in ``fetcher.py`` (descriptive
User-Agent with a contact URL, Retry-After-aware 429 backoff, on-disk
cache keyed by a stable id) so B-roll fetching behaves like the rest of
the pipeline. It does NOT copy code from any AGPL project.

Public API
----------
    fetch_broll(theme, count, cache_dir, used=None) -> list[Path]

Returns up to ``count`` distinct downloaded video paths. Best-effort:
network/rate-limit failures are logged and skipped, never raised, so a dead
source can't abandon a reel (same philosophy as fetcher.py's --single guard).

CLI (for smoke testing):
    python broll_fetcher.py "stoicism endurance" --count 3 --out ./cache/broll
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("broll")

# Whitelist sanitizer for search queries (project rule: allow only known-good
# values, never escape/strip "bad" chars). Search themes are free text that
# flows into a Wikimedia search string and an Archive.org Lucene query, both of
# which give special meaning to ":", "(", ")", quotes, AND/OR, etc. We keep only
# letters, digits, and single spaces -- losing operators is fine for a B-roll
# keyword search and removes any injection surface.
_ALLOWED_QUERY = re.compile(r"[^a-zA-Z0-9 ]+")


def _safe_query(theme: str) -> str:
    cleaned = _ALLOWED_QUERY.sub(" ", str(theme))
    cleaned = " ".join(cleaned.split())  # collapse runs of whitespace
    return cleaned[:120]                 # cap length; long quotes hurt recall anyway

# --- Polite-access conventions (mirrors fetcher.py) -------------------------
HEADERS = {
    "User-Agent": (
        "PhilosopherPipeline-BRoll/1.0 "
        "(+https://github.com/shauryalowkeygotaura/philosopher-pipeline; "
        "instagram-content-bot)"
    ),
    "Accept-Encoding": "gzip",
}
MAX_RETRIES_429 = 4
SEARCH_SLEEP = 0.5          # pre-sleep before search calls
INFO_SLEEP = 0.3           # pre-sleep before imageinfo calls
DOWNLOAD_SLEEP = 0.5       # pre-sleep before a clip download

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
ARCHIVE_SEARCH = "https://archive.org/advancedsearch.php"
ARCHIVE_META = "https://archive.org/metadata/"

# Container/codec we can feed straight into ffmpeg without surprises.
VIDEO_MIME_PREFIXES = ("video/webm", "video/ogg", "application/ogg", "video/mp4")
VIDEO_EXTS = (".webm", ".ogv", ".ogg", ".mp4", ".mpg", ".mpeg", ".mov")

# Background B-roll wants short, light clips -- skip multi-hundred-MB originals.
MAX_CLIP_BYTES = 60 * 1024 * 1024      # 60 MB hard ceiling per clip
PREFERRED_MAX_BYTES = 25 * 1024 * 1024  # rank smaller clips first


def _polite_get(url, params=None, timeout=30, pre_sleep=0.0):
    """GET with Retry-After-aware backoff for HTTP 429 (mirrors fetcher.py)."""
    if pre_sleep:
        time.sleep(pre_sleep)
    last_exc = None
    resp = None
    for attempt in range(MAX_RETRIES_429):
        resp = requests.get(url, params=params, timeout=timeout, headers=HEADERS)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        retry_after = resp.headers.get("Retry-After")
        try:
            wait = float(retry_after) if retry_after else 2.0 * (attempt + 1)
        except ValueError:
            wait = 2.0 * (attempt + 1)
        wait = min(wait, 30.0)
        log.info("HTTP 429 on %s; sleeping %.1fs (attempt %d/%d)",
                 url, wait, attempt + 1, MAX_RETRIES_429)
        time.sleep(wait)
        last_exc = requests.HTTPError("429 after retries", response=resp)
    if last_exc:
        raise last_exc
    return resp


def _clip_id(url: str) -> str:
    """Stable cache id from the URL (keeps the original extension)."""
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
    ext = next((e for e in VIDEO_EXTS if url.lower().split("?")[0].endswith(e)), ".mp4")
    return digest + ext


def _download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        resp = _polite_get(url, timeout=120, pre_sleep=DOWNLOAD_SLEEP)
        data = resp.content
        if len(data) > MAX_CLIP_BYTES:
            log.info("Skip oversized clip (%.1f MB > %d MB): %s",
                     len(data) / 1e6, MAX_CLIP_BYTES // (1024 * 1024), url)
            return False
        dest.write_bytes(data)
        return dest.stat().st_size > 0
    except Exception as e:  # noqa: BLE001 - best-effort, never abandon a reel
        log.warning("B-roll download failed (%s): %s", url, e)
        return False


# --- Wikimedia Commons (video) ---------------------------------------------
def _wikimedia_video_candidates(query: str, srlimit: int = 40) -> list[dict]:
    """Return [{url, size, mime, title}] of video files matching `query`."""
    query = _safe_query(query)
    if not query:
        return []
    params = {
        "action": "query", "format": "json",
        "generator": "search", "gsrnamespace": "6",
        "gsrsearch": f"filetype:video {query}", "gsrlimit": str(srlimit),
        "prop": "imageinfo", "iiprop": "url|size|mime",
    }
    try:
        resp = _polite_get(WIKIMEDIA_API, params=params, pre_sleep=SEARCH_SLEEP)
    except Exception as e:  # noqa: BLE001
        log.warning("Wikimedia video search failed for %r: %s", query, e)
        return []
    pages = resp.json().get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        for info in page.get("imageinfo", []) or []:
            mime = (info.get("mime") or "").lower()
            url = info.get("url")
            if not url or not mime.startswith(VIDEO_MIME_PREFIXES):
                continue
            out.append({
                "url": url,
                "size": int(info.get("size") or 0),
                "mime": mime,
                "title": page.get("title", ""),
                "source": "wikimedia",
            })
    return out


# --- Archive.org (movies) --------------------------------------------------
def _archive_video_candidates(query: str, rows: int = 8) -> list[dict]:
    query = _safe_query(query)
    if not query:
        return []
    params = {
        "q": f'({query}) AND mediatype:(movies)',
        "fl[]": "identifier",
        "rows": str(rows), "page": "1", "output": "json",
    }
    try:
        resp = _polite_get(ARCHIVE_SEARCH, params=params, pre_sleep=SEARCH_SLEEP)
        docs = resp.json().get("response", {}).get("docs", [])
    except Exception as e:  # noqa: BLE001
        log.warning("Archive.org search failed for %r: %s", query, e)
        return []

    out = []
    for doc in docs:
        ident = doc.get("identifier")
        if not ident:
            continue
        try:
            meta = _polite_get(ARCHIVE_META + ident, pre_sleep=INFO_SLEEP).json()
        except Exception as e:  # noqa: BLE001
            log.warning("Archive.org metadata failed for %s: %s", ident, e)
            continue
        server = meta.get("server")
        d = meta.get("dir")
        for f in meta.get("files", []) or []:
            name = f.get("name", "")
            if not name.lower().endswith(VIDEO_EXTS):
                continue
            if not (server and d):
                continue
            size = int(f.get("size") or 0)
            out.append({
                "url": f"https://{server}{d}/{requests.utils.quote(name)}",
                "size": size,
                "mime": "video/" + name.lower().rsplit(".", 1)[-1],
                "title": f"{ident}/{name}",
                "source": "archive_org",
            })
            break  # one representative clip per item is plenty
    return out


def _rank(candidates: list[dict]) -> list[dict]:
    """Prefer real, reasonably-sized clips.

    Heuristic background B-roll wants small/light files (faster download,
    less likely to be a feature-length original). Zero-size (unknown) sinks
    to the middle; anything over the hard ceiling is dropped upstream.
    """
    def key(c):
        size = c["size"] or PREFERRED_MAX_BYTES  # unknown -> treat as middling
        over = max(0, size - PREFERRED_MAX_BYTES)
        return over  # smaller-or-equal preferred sorts first
    return sorted([c for c in candidates if c["size"] <= MAX_CLIP_BYTES or c["size"] == 0], key=key)


def fetch_broll(
    theme: str,
    count: int,
    cache_dir,
    used: Optional[set] = None,
) -> list[Path]:
    """Fetch up to `count` distinct B-roll clips themed to `theme`.

    `used` is an optional set of already-used clip ids (filenames) to avoid
    repeats across a run; it is updated in place with whatever we return.
    Never raises on network failure -- returns as many clips as it got.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    used = used if used is not None else set()

    candidates = _wikimedia_video_candidates(theme)
    if len(candidates) < count * 2:
        candidates += _archive_video_candidates(theme)

    results: list[Path] = []
    seen_urls: set[str] = set()
    for cand in _rank(candidates):
        if len(results) >= count:
            break
        url = cand["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        dest = cache_dir / _clip_id(url)
        if dest.name in used:
            continue
        if _download(url, dest):
            used.add(dest.name)
            results.append(dest)
            log.info("B-roll [%s] %s -> %s (%.1f MB)",
                     cand["source"], cand["title"][:60], dest.name,
                     dest.stat().st_size / 1e6)
    if not results:
        log.warning("No B-roll fetched for theme %r", theme)
    return results


# --- Quote -> atmospheric theme ------------------------------------------
# Abstract philosophy ("we suffer more in imagination...") has no footage, but
# its imagery maps to atmospheric scenes that DO. Keyword groups map a quote to
# a concrete, search-friendly visual theme; anything unmatched rotates through a
# pool of universally-available atmospheric themes (seeded by the text so the
# same quote is stable but different quotes vary). Keyless, deterministic, no LLM.
_THEME_MAP = [
    (("death", "die", "mortal", "grave", "end", "dying"), "candle flame dark"),
    (("time", "moment", "hour", "fleeting", "transient", "river"), "flowing river water"),
    (("fear", "anxiety", "worry", "suffer", "pain", "dread"), "storm clouds sky"),
    (("sea", "ocean", "water", "wave", "tide", "storm"), "ocean waves"),
    (("light", "sun", "dawn", "hope", "bright", "shine"), "sunrise clouds sky"),
    (("night", "dark", "star", "shadow", "sleep"), "starry night sky"),
    (("mountain", "climb", "strength", "endure", "hard", "rock"), "mountain landscape"),
    (("fire", "burn", "passion", "flame", "desire"), "fire flames"),
    (("mind", "thought", "think", "soul", "reason", "wisdom"), "fog forest mist"),
    (("nature", "earth", "world", "life", "grow", "tree"), "forest trees nature"),
    (("war", "fight", "battle", "anger", "conflict"), "stormy sea waves"),
    (("calm", "peace", "still", "quiet", "rest"), "calm lake reflection"),
]
_THEME_POOL = [
    "ocean waves", "storm clouds sky", "mountain landscape", "starry night sky",
    "forest fog mist", "sunrise clouds", "rain on glass", "desert dunes",
]


def derive_broll_theme(quote: str, philosopher: str = "") -> str:
    """Map a quote to a concrete atmospheric B-roll search theme."""
    text = (quote or "").lower()
    for keys, theme in _THEME_MAP:
        if any(k in text for k in keys):
            return theme
    seed = sum(ord(c) for c in (text + (philosopher or "").lower())) % len(_THEME_POOL)
    return _THEME_POOL[seed]


def _cli() -> int:
    ap = argparse.ArgumentParser(description="Fetch keyless video B-roll.")
    ap.add_argument("theme", help="Search theme, e.g. 'stoicism endurance'")
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--out", default="./cache/broll")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    clips = fetch_broll(args.theme, args.count, args.out)
    print(f"\nFetched {len(clips)} clip(s):")
    for c in clips:
        print(f"  {c}  ({c.stat().st_size / 1e6:.1f} MB)")
    return 0 if clips else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
