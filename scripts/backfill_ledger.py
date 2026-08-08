"""
backfill_ledger.py -- seed the reward loop from reels already on Instagram.

WHY
---
uploader.py has been writing ledger rows since Phase 1, but the CI workflow
never committed runs/upload_ledger.jsonl back to the repo, so every row died
with its ephemeral runner. After 109 posts the ledger did not exist and both
bandits (caption hook, quote theme) were pinned to their round-robin baseline.

The engagement data itself is not lost -- it is on the published media. This
script walks the account's reels, reconstructs a ledger row per post, and
attaches live insights, giving the bandits a real cold-start prior instead of
30-50 posts of waiting.

HOW A ROW IS RECONSTRUCTED
--------------------------
  media_id   <- media.pk
  quote      <- parsed from the caption, which _build_caption() writes as
                  <hook>\\n"<quote>"\\n- <Philosopher>
  philosopher<- the "- <Name>" attribution line, validated against
                philosophers.md so a stray caption cannot invent an arm
  theme      <- quotes.classify_theme() (Groq, keyword fallback)
  hook       <- matched against pipeline.HOOKS by exact prefix
  insights   <- insights_media if the account is Business/Creator, else the
                public like/comment counts on the media object

Idempotent: media ids already present in the ledger are skipped, so re-running
after a partial failure is safe.

USAGE
-----
    doppler run -- python scripts/backfill_ledger.py --dry-run
    doppler run -- python scripts/backfill_ledger.py --limit 120
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ledger  # noqa: E402
import quotes  # noqa: E402
from input_parser import parse_philosophers  # noqa: E402

log = logging.getLogger("backfill_ledger")

_ROOT = Path(__file__).resolve().parent.parent

# _build_caption() emits: <hook>\n"<quote>"\n- <Philosopher>\n\n<bio>...
# Anchor on the attribution line so a bio containing quote marks cannot fool us.
_CAPTION_RE = re.compile(
    r'"(?P<quote>[^"]{15,300})"\s*\n\s*[-–—]\s*(?P<philosopher>[^\n#]{3,40})',
    re.MULTILINE,
)


def parse_caption(caption: str, known: set[str]) -> tuple[str, str] | None:
    """Extract (quote, philosopher) from a published caption.

    `known` is the whitelist from philosophers.md. We VALIDATE the parsed name
    against it rather than trusting whatever the regex caught: an unrecognized
    name means we failed to parse, not that a new philosopher exists.
    """
    if not isinstance(caption, str):
        return None
    m = _CAPTION_RE.search(caption)
    if not m:
        return None
    quote = m.group("quote").strip()
    name = m.group("philosopher").strip()
    lookup = {k.lower(): k for k in known}
    canonical = lookup.get(name.lower())
    if canonical is None:
        log.debug("caption attributes to unknown name %r; skipping", name)
        return None
    return quote, canonical


def public_metrics(media) -> dict:
    """Like/comment counts straight off the media object.

    Fallback reward source when insights_media is unavailable (personal
    account, or the endpoint refuses). bandit.reward() reads these keys.
    """
    out = {}
    for attr, key in (("like_count", "like_count"), ("comment_count", "comment_count")):
        value = getattr(media, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            out[key] = value
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    ap.add_argument("--limit", type=int, default=200, help="max reels to walk")
    ap.add_argument("--no-insights", action="store_true",
                    help="skip the account-gated insights call, use public counts only")
    ap.add_argument("--sleep", type=float, default=1.5,
                    help="seconds between per-media calls (IG rate limiting)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    known = set(parse_philosophers(_ROOT / "philosophers.md"))
    if not known:
        log.error("philosophers.md yielded no names; aborting.")
        return 1

    existing = {r.get("media_id") for r in ledger.load_entries()}
    log.info("Ledger currently holds %d rows.", len(existing))

    try:
        import uploader
        client = uploader._get_client()
    except Exception as e:  # noqa: BLE001
        log.error("Instagram login failed: %s", e)
        log.error("Backfill needs a working session. Check INSTAGRAM_USERNAME / "
                  "INSTAGRAM_PASSWORD in Doppler, and clear any checkpoint by "
                  "logging in from a browser once.")
        return 1

    try:
        user_id = client.user_id
        medias = client.user_clips(user_id, amount=args.limit)
    except Exception as e:  # noqa: BLE001
        log.error("Could not list reels: %s", e)
        return 1

    log.info("Fetched %d reels from the account.", len(medias))

    written = skipped = unparsed = 0
    for media in medias:
        media_id = ledger.extract_media_id(media)
        if media_id is None:
            unparsed += 1
            continue
        if media_id in existing:
            skipped += 1
            continue

        parsed = parse_caption(getattr(media, "caption_text", "") or "", known)
        if parsed is None:
            unparsed += 1
            continue
        quote, philosopher = parsed

        metrics = {}
        if not args.no_insights:
            try:
                import insights as insights_mod
                pulled = insights_mod.pull_one(client, media_id)
                if pulled:
                    metrics = pulled
            except Exception as e:  # noqa: BLE001
                log.debug("insights unavailable for %s: %s", media_id, e)
        if not metrics:
            metrics = public_metrics(media)

        if not metrics:
            log.debug("no reward signal for %s; recording without insights", media_id)

        theme = quotes.classify_theme(quote, philosopher)
        reward = quotes.bandit.reward(metrics)
        log.info(
            "%s  %-20s %-12s reward=%s  %.50s",
            media_id, philosopher, theme,
            f"{reward:.0f}" if reward is not None else "none", quote,
        )

        if not args.dry_run:
            recorded = ledger.record_upload(
                media_id,
                mp4_path=None,
                caption=getattr(media, "caption_text", None),
                philosopher=philosopher,
                quote=quote,
                theme=theme,
                style="kinetic",
                extra={"backfilled": True},
            )
            if recorded and metrics:
                ledger.attach_insights(recorded, metrics)
        written += 1
        time.sleep(args.sleep)

    log.info(
        "Done. %d rows %s, %d already present, %d captions unparsed.",
        written, "would be written (--dry-run)" if args.dry_run else "written",
        skipped, unparsed,
    )

    if not args.dry_run and written:
        stats = quotes.theme_stats()
        log.info("Theme priors now seeded:")
        for theme in quotes.THEMES:
            s = stats.get(theme)
            log.info(
                "  %-14s n=%-4s mean=%s",
                theme,
                int(s["n"]) if s else 0,
                f"{s['mean']:.1f}" if s else "-",
            )
        log.info("Commit runs/upload_ledger.jsonl so CI inherits these priors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
