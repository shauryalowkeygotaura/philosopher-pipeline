"""
backfill_state.py -- one-shot repair of state.json's duplicated history.

WHY
---
fetch_quote() used to return PHILOSOPHER_QUOTES[0] on every run once a
philosopher's quotes were spent, and update_philosopher() appended that same
string to used_quotes each time. Albert Camus therefore has "In the midst of
winter..." recorded three times. Separately, dedup compared RAW strings, so
punctuation variants of one quote ("walk in front of me - I may not follow" vs
"walk in front of me, I may not lead") were stored as two distinct entries.

The result is a used_quotes list that overstates the real history and hides how
badly the pool was exhausted. This collapses each list to canonical-unique
entries, preserving first-seen order.

post_count is NOT touched: it records how many reels actually shipped, which is
true regardless of how many distinct quotes they used. The gap between
post_count and len(used_quotes) after this runs is precisely the number of
duplicate publishes the frozen-[0] bug caused.

USAGE
-----
    python scripts/backfill_state.py --dry-run    # report only
    python scripts/backfill_state.py              # write (backs up first)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quotes import canon  # noqa: E402

STATE_PATH = Path(__file__).resolve().parent.parent / "state.json"

# used_songs / used_photos are plain identifiers (URLs, filenames) where exact
# string equality is already the right notion of identity; only quotes need the
# canonical-form collapse.
_LIST_FIELDS = ("used_quotes", "used_songs", "used_photos")


def _dedupe(values: list, *, canonical: bool) -> list:
    """First-seen-order unique. `canonical` applies quote normalization."""
    seen: set = set()
    out: list = []
    for v in values:
        if not isinstance(v, str):
            continue
        key = canon(v) if canonical else v.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    ap.add_argument("--path", default=str(STATE_PATH), help="state.json path")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: {path} does not exist.", file=sys.stderr)
        return 1

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: {path} is not valid JSON: {e}", file=sys.stderr)
        return 1

    changed = False
    print(f"{'philosopher':22s} {'posts':>5s} {'quotes':>14s} {'songs':>12s} {'photos':>12s}")
    print("-" * 70)

    for name, entry in data.items():
        if name.startswith("_") or not isinstance(entry, dict):
            continue
        report = []
        for field in _LIST_FIELDS:
            original = entry.get(field)
            if not isinstance(original, list):
                report.append("n/a")
                continue
            cleaned = _dedupe(original, canonical=(field == "used_quotes"))
            if len(cleaned) != len(original):
                changed = True
            entry[field] = cleaned
            report.append(f"{len(original)}->{len(cleaned)}")
        print(
            f"{name:22s} {entry.get('post_count', 0):5d} "
            f"{report[0]:>14s} {report[1]:>12s} {report[2]:>12s}"
        )

    if not changed:
        print("\nNo duplicates found; state.json already clean.")
        return 0

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"state.json.bak.{stamp}")
    shutil.copy(path, backup)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    print(f"\nWrote {path} (backup: {backup.name}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
