"""
uploader.py — Real instagrapi 2.3.0 Reel uploader.

Public API:
    upload_reel(mp4_path: str, caption: str) -> True

Session is cached to <username>.session in the pipeline directory so
we don't re-login on every run. The instagrapi Client is created lazily
on the first call to upload_reel().
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Pipeline root (directory this file lives in) ──────────────────────────────
_PIPELINE_DIR = Path(__file__).parent.resolve()

# ── Module-level singleton (populated on first upload_reel call) ───────────────
_client = None


# ── .env fallback loader (python-dotenv not required) ─────────────────────────
def _load_dotenv() -> None:
    """Parse a .env file in the pipeline directory and inject into os.environ.
    Variables already set in the environment are NOT overwritten (dotenv behaviour).
    """
    env_path = _PIPELINE_DIR / ".env"
    if not env_path.exists():
        return
    with env_path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


# ── Client factory ─────────────────────────────────────────────────────────────
def _get_client():
    """Return the lazily-initialised, logged-in instagrapi Client singleton."""
    global _client
    if _client is not None:
        return _client

    # Try to load credentials from env (populate from .env first)
    _load_dotenv()
    username = os.environ.get("INSTAGRAM_USERNAME", "").strip()
    password = os.environ.get("INSTAGRAM_PASSWORD", "").strip()
    if not username or not password:
        raise ValueError(
            "INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD must be set "
            "as environment variables or in the .env file."
        )

    from instagrapi import Client  # deferred import — keeps module-import fast

    cl = Client()
    session_path = _PIPELINE_DIR / f"{username}.session"

    # Load cached session if available; fall back to fresh login if it is stale
    if session_path.exists():
        try:
            cl.load_settings(session_path)
            cl.login(username, password)  # refreshes token using cached cookies
        except Exception:
            # Session expired or corrupt — do a clean login
            cl = Client()
            cl.login(username, password)
    else:
        cl.login(username, password)

    # Persist session so subsequent runs skip full re-login
    cl.dump_settings(session_path)

    _client = cl
    return _client


# ── Public API ─────────────────────────────────────────────────────────────────
def upload_reel(
    mp4_path: str,
    caption: str,
    jpg_path: str | None = None,
    meta: dict | None = None,
) -> bool:
    """Upload a Reel to Instagram.

    Args:
        mp4_path: Absolute path to the MP4 file.
        caption:  Post caption / description.
        jpg_path: Optional path to a .jpg thumbnail image.
        meta:     Optional dict of arm metadata (philosopher / hook / slogan /
                  slug / style) recorded alongside the posted media id so the
                  self-improving bandit loop can attribute later insights back
                  to the caption hook that earned them. Purely additive; has no
                  effect on the upload itself.

    Returns:
        True on success.

    Raises:
        ValueError: If credentials are missing.
        Exception:  Any instagrapi error (pipeline handles retry).
    """
    cl = _get_client()
    thumbnail = Path(jpg_path) if jpg_path and Path(jpg_path).exists() else None
    media = cl.clip_upload(Path(mp4_path), caption=caption, thumbnail=thumbnail)
    _record_to_ledger(media, mp4_path, caption, meta)
    return True


def _record_to_ledger(media, mp4_path: str, caption: str, meta: dict | None) -> None:
    """Capture the posted media id (Phase 1 of the self-improving loop).

    Best-effort and fully isolated: a ledger/import failure, or a media object
    without a valid numeric id (e.g. a test double), is swallowed so it can
    never affect the upload result. Only a whitelist-validated id is stored.
    """
    try:
        import ledger as _ledger  # deferred so module import stays dependency-free

        meta = meta or {}
        _ledger.record_upload(
            media,
            mp4_path=mp4_path,
            caption=caption,
            philosopher=meta.get("philosopher"),
            hook=meta.get("hook"),
            slogan=meta.get("slogan"),
            slug=meta.get("slug"),
            style=meta.get("style"),
        )
    except Exception:
        return
