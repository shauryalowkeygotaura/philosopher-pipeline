"""
topup_pool.py -- extend runs/quote_pool.json ahead of demand.

WHY
---
select_quote() can top up on the fly, but doing it inside a reel run means a
cron slot pays two Groq round trips (recall + attribution verify) and, if both
come back empty, the reel ships an LRU replay. Running this out of band keeps a
buffer of verified quotes so the live path almost always finds something fresh.

YIELD IS LOW BY DESIGN
----------------------
verify_quotes() fails closed and is deliberately over-strict. Measured against
18 known-genuine and 8 known-fabricated quotes, the shipped configuration scores
~50% recall at 0% false-accept. Loosening the prompt to reach 61% recall let a
fabrication through, so recall was traded away on purpose: a false accept
publishes an invented quotation under a real person's name, while a false reject
costs one extra API call.

That is why this asks for 12 candidates across several themes per philosopher.
Yield comes from volume, never from relaxing the gate.

USAGE
-----
    doppler run -- python scripts/topup_pool.py --dry-run
    doppler run -- python scripts/topup_pool.py --target 12
    doppler run -- python scripts/topup_pool.py --philosopher "Franz Kafka"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import quotes  # noqa: E402
from fetcher import PHILOSOPHER_QUOTES  # noqa: E402
from input_parser import parse_philosophers  # noqa: E402

log = logging.getLogger("topup_pool")
_ROOT = Path(__file__).resolve().parent.parent


def unused_count(philosopher: str, state: dict) -> int:
    """How many pool quotes this philosopher has never published."""
    key = philosopher.lower()
    pool = [
        {"quote": q} for q in PHILOSOPHER_QUOTES.get(key, [])
    ] + quotes.load_pool().get(key, [])
    used = quotes.canon_set(
        state.get(philosopher, {}).get("used_quotes", [])
    )
    return sum(1 for r in pool if quotes.canon(r["quote"]) not in used)


def prune(names: list[str], *, dry_run: bool, sleep: float) -> int:
    """Re-run the attribution check over quotes already in the pool.

    Needed whenever the verifier is tightened: the 2026-08-01 top-up ran before
    truncation detection existed and admitted fragments like "You must have
    chaos within you". Those are already persisted, so re-verification is the
    only way to get them back out.
    """
    if not os.environ.get("GROQ_API_KEY"):
        log.error("GROQ_API_KEY is not set. Prune DELETES on a negative verdict "
                  "and verify_quotes() cannot run without a key -- refusing to "
                  "continue. Re-run under: doppler run -- python %s --prune",
                  Path(__file__).name)
        return 1

    pool = quotes.load_pool()
    if not pool:
        log.info("pool is empty; nothing to prune.")
        return 0

    # Only count what this invocation actually inspects, so --philosopher does
    # not report a denominator covering the whole pool.
    visited = [p for p in names if pool.get(p.lower())]
    total_before = sum(len(pool[p.lower()]) for p in visited)
    if not total_before:
        log.info("none of the named philosophers have pooled quotes.")
        return 0

    removed_total = 0
    skipped: list[str] = []
    for philosopher in visited:
        key = philosopher.lower()
        rows = pool[key]
        try:
            # fail_closed=False: a 429 or malformed reply must RAISE here, not
            # read as "every quote is fabricated" and delete the lot.
            verdicts = quotes.verify_quotes(
                philosopher, [r["quote"] for r in rows], fail_closed=False,
            )
        except quotes.VerificationUnavailable as e:
            log.error("verification unavailable for %s (%s); aborting prune "
                      "with NOTHING written.", philosopher, e)
            return 1

        keep = [r for r, ok in zip(rows, verdicts) if ok]
        drop = [r for r, ok in zip(rows, verdicts) if not ok]

        # A 100% rejection is far more likely to be a degraded model response
        # than a philosopher whose every pooled quote is fake. SKIP this entry
        # rather than empty it -- and keep going, so one suspect philosopher
        # does not prevent the rest of the pool from being cleaned.
        if rows and not keep:
            log.warning("%s: verifier rejected ALL %d quotes. That is "
                        "implausible; leaving this philosopher UNTOUCHED.",
                        philosopher, len(rows))
            skipped.append(philosopher)
            time.sleep(sleep)
            continue

        for r in drop:
            log.info("  DROP %-22s %s", philosopher, r["quote"][:70])
        pool[key] = keep
        removed_total += len(drop)
        log.info("%-22s %d -> %d", philosopher, len(rows), len(keep))
        time.sleep(sleep)

    # Skipped philosophers were never really inspected, so exclude them from
    # the denominator rather than overstating coverage.
    inspected = total_before - sum(len(pool[p.lower()]) for p in skipped)
    if skipped:
        log.warning("left untouched (all-reject, likely degraded verifier): %s",
                    ", ".join(skipped))

    if dry_run:
        log.info("")
        log.info("--dry-run: would drop %d of %d inspected.", removed_total, inspected)
        return 0

    if removed_total:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = quotes.QUOTE_POOL_PATH.with_name(f"quote_pool.json.bak.{stamp}")
        try:
            shutil.copy(quotes.QUOTE_POOL_PATH, backup)
            log.info("backup written: %s", backup.name)
        except OSError as e:
            log.error("could not back up the pool (%s); aborting.", e)
            return 1

    quotes.save_pool({k: v for k, v in pool.items() if v})
    log.info("")
    log.info("Pruned %d of %d inspected; %d remain.",
             removed_total, total_before, total_before - removed_total)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=8,
                    help="desired unused quotes per philosopher")
    ap.add_argument("--philosopher", action="append",
                    help="limit to these names (repeatable)")
    ap.add_argument("--dry-run", action="store_true",
                    help="generate and report without writing the pool")
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="seconds between Groq calls")
    ap.add_argument("--prune", action="store_true",
                    help="re-verify quotes ALREADY in the pool and drop failures, "
                         "then exit (use after tightening the verifier)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "groq", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        state = json.loads((_ROOT / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as e:
        log.error("cannot read state.json: %s", e)
        return 1

    names = args.philosopher or parse_philosophers(_ROOT / "philosophers.md")
    if not names:
        log.error("no philosophers to process.")
        return 1

    if args.prune:
        return prune(names, dry_run=args.dry_run, sleep=args.sleep)

    grand_total = 0
    for philosopher in names:
        have = unused_count(philosopher, state)
        if have >= args.target:
            log.info("%-22s %d unused, at target; skipping.", philosopher, have)
            continue

        used = quotes.canon_set(state.get(philosopher, {}).get("used_quotes", []))
        added_here = 0

        # Rotate themes so the buffer is spread across the bandit's arm space
        # rather than piling up on whichever theme happens to generate easily.
        for theme in quotes.THEMES:
            if have + added_here >= args.target:
                break
            pool_now = quotes.load_pool().get(philosopher.lower(), [])
            avoid = used | quotes.canon_set(
                list(PHILOSOPHER_QUOTES.get(philosopher.lower(), []))
                + [r["quote"] for r in pool_now]
            ) | quotes.known_elsewhere(philosopher)
            generated = quotes.generate_quotes(
                philosopher, theme=theme, avoid=avoid, n=12,
            )
            if generated:
                for row in generated:
                    log.info("    + [%s] %s", theme, row["quote"][:72])
                if not args.dry_run:
                    added_here += quotes.add_to_pool(philosopher, generated)
                else:
                    added_here += len(generated)
            time.sleep(args.sleep)

        log.info("%-22s %d unused -> %d (+%d)",
                 philosopher, have, have + added_here, added_here)
        grand_total += added_here

    log.info("")
    log.info("Added %d quotes%s.", grand_total,
             " (dry run, nothing written)" if args.dry_run else "")
    if grand_total and not args.dry_run:
        log.info("Commit runs/quote_pool.json so CI inherits the buffer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
