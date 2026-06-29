"""
insights.py -- Phase 3 (live) insights pull for the self-improving reel loop.

This is the ONLY part of the loop that talks to Instagram for analytics, and it
is doubly gated:
  1. bandit.live_insights_enabled()  -> env flag PHILOSOPHER_LIVE_INSIGHTS, OFF by default.
  2. account type                    -> the instagrapi insights endpoint only
                                        works for Business / Creator accounts.

Nothing here is auto-invoked by the pipeline. Run it explicitly (e.g. a manual
maintenance pass or a scheduled job) once posts have aged enough for Instagram
to have computed their insights:

    doppler run -- python -c "import insights; insights.refresh_pending()"

Every external response is status/shape-checked before use; any failure on one
media id is logged and skipped so a single bad id never aborts the batch.
"""
from __future__ import annotations

import logging
from typing import Any

import bandit
import ledger

log = logging.getLogger(__name__)


def _coerce_insights(raw: Any) -> dict[str, Any] | None:
    """Validate the instagrapi insights response into a plain dict, or None."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    # instagrapi may hand back a pydantic-ish object; only read, never call.
    for attr in ("dict", "model_dump"):
        fn = getattr(raw, attr, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, dict):
                    return out
            except Exception:
                continue
    return None


def pull_one(client: Any, media_id: str) -> dict[str, Any] | None:
    """Pull insights for a single media id via instagrapi. Returns a dict or None.

    `client` is a logged-in instagrapi Client (insights_media is account-gated).
    All errors are caught: a private/ineligible account or a transient network
    error yields None instead of raising.
    """
    valid = ledger.validate_media_id(media_id)
    if valid is None:
        return None
    pk = valid.split("_", 1)[0]  # insights_media expects the numeric pk
    try:
        raw = client.insights_media(pk)
    except Exception as e:  # noqa: BLE001 - external boundary, never propagate
        log.warning("insights pull failed for %s: %s", valid, e)
        return None
    return _coerce_insights(raw)


def refresh_pending(client: Any | None = None, limit: int | None = None) -> int:
    """Pull and attach insights for ledger rows that still lack them.

    No-op (returns 0) unless the Phase 3 flag is on. Builds an instagrapi client
    lazily only when actually enabled, so importing/calling this with the flag
    off makes zero network calls. Returns the number of rows updated.
    """
    if not bandit.live_insights_enabled():
        log.info("refresh_pending: PHILOSOPHER_LIVE_INSIGHTS off; skipping live pull.")
        return 0

    pending = ledger.pending_insight_ids()
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        return 0

    if client is None:
        # Reuse the uploader's logged-in singleton so we do not re-auth.
        import uploader
        client = uploader._get_client()

    updated = 0
    for media_id in pending:
        data = pull_one(client, media_id)
        if data and ledger.attach_insights(media_id, data):
            updated += 1
    log.info("refresh_pending: attached insights to %d/%d rows.", updated, len(pending))
    return updated
