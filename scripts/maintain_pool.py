"""
maintain_pool.py -- the pipeline keeps its own quote supply stocked.

WHAT THIS REPLACES
------------------
Running scripts/topup_pool.py by hand every few weeks. This is the unattended
version: CI runs it on a schedule, it decides what the pool needs, and it
commits the result. Nobody has to notice the pool is draining.

WHAT IT DOES, IN ORDER
----------------------
1. Measure runway: how many unpublished quotes each active philosopher has.
2. Top up anyone below --target, asking Groq for candidates across themes.
3. If TOTAL runway is still below --min-runway, promote the next philosopher
   from roster.py and stock them. A new thinker is the only unbounded source
   of genuine quotes; topping up the same 12 forever hits a hard ceiling.
4. Emit runs/pool_status.json so the workflow can open an issue when the pool
   is starving and no amount of generation is fixing it.

Promotion is capped per run (--max-promotions, default 1) so a single bad run
cannot dump the whole bench onto the account.

WHY IT NEVER DELETES
--------------------
Pruning is destructive and stays manual (topup_pool.py --prune). An unattended
job that can delete the pool on a bad model day is exactly the failure this
codebase already learned about the hard way.

USAGE
-----
    doppler run -- python scripts/maintain_pool.py --dry-run
    doppler run -- python scripts/maintain_pool.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import quotes  # noqa: E402
import roster  # noqa: E402
from fetcher import PHILOSOPHER_QUOTES  # noqa: E402
from input_parser import parse_philosophers  # noqa: E402

log = logging.getLogger("maintain_pool")
_ROOT = Path(__file__).resolve().parent.parent
STATUS_PATH = _ROOT / "runs" / "pool_status.json"


def runway(philosopher: str, state: dict, pool: dict) -> int:
    """Unpublished quotes available to this philosopher right now."""
    key = philosopher.lower()
    rows = [{"quote": q} for q in PHILOSOPHER_QUOTES.get(key, [])] + pool.get(key, [])
    used = quotes.canon_set(state.get(philosopher, {}).get("used_quotes", []))
    return sum(1 for r in rows if quotes.canon(r["quote"]) not in used)


def stock(philosopher: str, state: dict, *, target: int, sleep: float,
          dry_run: bool) -> int:
    """Generate until `philosopher` has `target` unpublished quotes.

    Rotates themes so the buffer spreads across the bandit's arm space instead
    of piling onto whichever theme happens to generate easily. Returns the
    number of quotes actually added.
    """
    added = 0
    used = quotes.canon_set(state.get(philosopher, {}).get("used_quotes", []))
    for theme in quotes.THEMES:
        have = runway(philosopher, state, quotes.load_pool()) + (added if dry_run else 0)
        if have >= target:
            break
        pool_now = quotes.load_pool().get(philosopher.lower(), [])
        avoid = (
            used
            | quotes.canon_set(
                list(PHILOSOPHER_QUOTES.get(philosopher.lower(), []))
                + [r["quote"] for r in pool_now]
            )
            | quotes.known_elsewhere(philosopher)
        )
        try:
            generated = quotes.generate_quotes(
                philosopher, theme=theme, avoid=avoid, n=12,
            )
        except Exception as e:  # noqa: BLE001 - never let one theme abort the run
            log.warning("  %s/%s generation failed: %s", philosopher, theme, e)
            generated = []
        for row in generated:
            log.info("    + [%s] %s", theme, row["quote"][:70])
        if generated and not dry_run:
            added += quotes.add_to_pool(philosopher, generated)
        elif generated:
            added += len(generated)
        time.sleep(sleep)
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=8,
                    help="desired unpublished quotes per philosopher")
    ap.add_argument("--min-runway", type=int, default=40,
                    help="promote a new philosopher when TOTAL runway is below this")
    ap.add_argument("--max-promotions", type=int, default=1,
                    help="cap on philosophers promoted in a single run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "groq", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not os.environ.get("GROQ_API_KEY"):
        log.error("GROQ_API_KEY is not set; cannot generate quotes. "
                  "Run under: doppler run -- python scripts/maintain_pool.py")
        return 1

    try:
        state = json.loads((_ROOT / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as e:
        log.error("cannot read state.json: %s", e)
        return 1

    active = parse_philosophers(_ROOT / "philosophers.md")
    added_total = 0
    promoted: list[str] = []

    # --- 1/2. measure and top up ------------------------------------------
    for philosopher in active:
        have = runway(philosopher, state, quotes.load_pool())
        if have >= args.target:
            log.info("%-24s runway=%-3d ok", philosopher, have)
            continue
        log.info("%-24s runway=%-3d topping up...", philosopher, have)
        got = stock(philosopher, state, target=args.target,
                    sleep=args.sleep, dry_run=args.dry_run)
        added_total += got
        log.info("%-24s runway=%-3d (+%d)", philosopher,
                 runway(philosopher, state, quotes.load_pool()), got)

    pool = quotes.load_pool()
    total = sum(runway(p, state, pool) for p in active)
    log.info("")
    log.info("total runway across %d philosophers: %d (min %d)",
             len(active), total, args.min_runway)

    # --- 3. promote when the existing roster cannot carry the load --------
    while (total < args.min_runway
           and len(promoted) < args.max_promotions):
        bench = roster.available(active)
        if not bench:
            log.error("roster bench is EMPTY and runway is %d. Add candidates "
                      "to roster.CANDIDATES.", total)
            break
        pick = bench[0]
        log.info("runway %d < %d: promoting %s", total, args.min_runway, pick)
        if args.dry_run:
            log.info("  --dry-run: would promote %s", pick)
            promoted.append(pick)
            break
        if not roster.promote(pick):
            log.error("could not promote %s; stopping promotion.", pick)
            break
        promoted.append(pick)
        active.append(pick)
        got = stock(pick, state, target=args.target,
                    sleep=args.sleep, dry_run=False)
        added_total += got
        log.info("%-24s stocked with %d quotes", pick, got)
        pool = quotes.load_pool()
        total = sum(runway(p, state, pool) for p in active)

    # --- 4. status for the workflow --------------------------------------
    pool = quotes.load_pool()
    per_philosopher = {p: runway(p, state, pool) for p in active}
    starved = sorted(p for p, n in per_philosopher.items() if n <= 0)
    status = {
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_runway": sum(per_philosopher.values()),
        "min_runway": args.min_runway,
        "quotes_added": added_total,
        "promoted": promoted,
        "starved": starved,
        "per_philosopher": per_philosopher,
    }
    if not args.dry_run:
        try:
            STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATUS_PATH.write_text(
                json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            log.warning("could not write %s: %s", STATUS_PATH, e)

    log.info("")
    log.info("added %d quotes; promoted %s; runway %d; starved %s",
             added_total, promoted or "nobody", status["total_runway"],
             starved or "nobody")
    if starved:
        # Non-fatal: the LRU floor still ships a reel. The workflow turns this
        # into a GitHub issue so a starving pool cannot go unnoticed the way
        # the original frozen-quote bug did for 85 reels.
        log.error("STARVED (will republish via LRU replay): %s", ", ".join(starved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
