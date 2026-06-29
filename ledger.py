"""
ledger.py -- local self-improving reel loop, Phase 1-2 storage.

Records every posted reel (Instagram media id + the caption hook/slogan arm
that was chosen) to a local JSONL ledger so a bandit can later attribute
engagement insights back to the arm that earned them.

FREE and offline by design: no network call, no secret, no key. This module
only reads/writes a local file.

Phases of the loop:
  Phase 1  capture the posted media id + arm at upload time          (this file)
  Phase 2  a delayed insights pull writes engagement metrics back
           onto the matching ledger row; bandit.py reads those       (this file + bandit.py)
  Phase 3  the live insights pull itself is gated on a Business /
           Creator account and an explicit env flag                  (insights.py)

Until reward data accrues the bandit returns the legacy deterministic pick, so
reel output stays byte-identical.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DIR = Path(__file__).parent.resolve()

# runs/ is tracked (output/ and cache/ are gitignored), so the ledger survives
# across runs and machines without carrying any secret.
LEDGER_PATH = _DIR / "runs" / "upload_ledger.jsonl"

# Whitelist for an Instagram media identifier. pk is all digits; the full id is
# "<pk>_<user_id>". We VALIDATE against this shape rather than stripping bad
# characters: anything that does not match is treated as "no media id" and
# simply not recorded (never sanitized-then-stored).
_MEDIA_ID_RE = re.compile(r"^[0-9]+(_[0-9]+)?$")


def validate_media_id(value: Any) -> str | None:
    """Return the canonical media id string if it matches the whitelist, else None."""
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text if _MEDIA_ID_RE.match(text) else None


def extract_media_id(media: Any) -> str | None:
    """Pull a validated media id out of an instagrapi Media object or a dict.

    Prefers the short numeric `pk`; falls back to the full `<pk>_<user>` id.
    Only the `pk`/`id` attributes (or dict keys) are read, never called, so a
    test double / unexpected object can never trigger side effects. Anything
    that fails the whitelist returns None.
    """
    if media is None:
        return None

    candidates: list[Any] = []
    if isinstance(media, dict):
        candidates = [media.get("pk"), media.get("id")]
    else:
        candidates = [getattr(media, "pk", None), getattr(media, "id", None)]

    for candidate in candidates:
        valid = validate_media_id(candidate)
        if valid is not None:
            return valid
    return None


def record_upload(
    media: Any,
    *,
    mp4_path: str | None = None,
    caption: str | None = None,
    philosopher: str | None = None,
    hook: str | None = None,
    slogan: str | None = None,
    slug: str | None = None,
    style: str | None = None,
    extra: dict[str, Any] | None = None,
    path: Path | None = None,
) -> str | None:
    """Append one upload row to the ledger.

    `media` may be the instagrapi Media object returned by clip_upload, a dict,
    or a bare id string. If no valid media id can be extracted the row is NOT
    written (we never store junk ids) and None is returned. Best-effort: any IO
    error is swallowed so a ledger problem can never fail a real upload.

    Returns the recorded media id on success, else None.
    """
    media_id = media if isinstance(media, str) else None
    if media_id is not None:
        media_id = validate_media_id(media_id)
    if media_id is None:
        media_id = extract_media_id(media)
    if media_id is None:
        return None

    row: dict[str, Any] = {
        "media_id": media_id,
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "philosopher": philosopher,
        "hook": hook,
        "slogan": slogan,
        "slug": slug,
        "style": style,
        "mp4": Path(mp4_path).name if mp4_path else None,
        "insights": None,  # filled later by a delayed/Phase-3 insights pull
    }
    if extra:
        row["extra"] = extra

    target = Path(path) if path is not None else LEDGER_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        # Ledger is an enhancement, never a hard dependency of uploading.
        return None
    return media_id


def load_entries(path: Path | None = None) -> list[dict[str, Any]]:
    """Read every well-formed JSONL row from the ledger. Malformed lines are skipped."""
    target = Path(path) if path is not None else LEDGER_PATH
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    except Exception:
        return rows
    return rows


def attach_insights(media_id: str, insights: dict[str, Any], path: Path | None = None) -> bool:
    """Write an insights dict onto the most recent ledger row for `media_id`.

    Rewrites the JSONL file in place (the ledger is small: one row per post).
    Best-effort; returns True if a matching row was updated. The media id is
    validated first so a bad caller value can never create a phantom row.
    """
    valid = validate_media_id(media_id)
    if valid is None or not isinstance(insights, dict):
        return False

    target = Path(path) if path is not None else LEDGER_PATH
    rows = load_entries(target)
    if not rows:
        return False

    updated = False
    # Walk newest-first so the latest post for an id wins, but only flip one row.
    for row in reversed(rows):
        if row.get("media_id") == valid:
            row["insights"] = insights
            updated = True
            break
    if not updated:
        return False

    try:
        with target.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        return False
    return True


def pending_insight_ids(path: Path | None = None) -> list[str]:
    """Return media ids that have been posted but have no insights attached yet."""
    out: list[str] = []
    for row in load_entries(path):
        mid = row.get("media_id")
        if mid and not row.get("insights"):
            out.append(mid)
    return out
