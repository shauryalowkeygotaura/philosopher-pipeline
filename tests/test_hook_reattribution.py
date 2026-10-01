"""Backfilled rows must be creditable to a hook, or the bandit never learns."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import ledger  # noqa: E402
from backfill_ledger import match_hook  # noqa: E402

HOOKS = ["the kind of words that hit at 3am.", "screenshot this one."]


def test_match_hook_exact_first_line():
    cap = 'the kind of words that hit at 3am.\n"Some quote here ok"\n- Seneca'
    assert match_hook(cap, HOOKS) == HOOKS[0]


def test_match_hook_ignores_case_whitespace_and_trailing_period():
    assert match_hook("\n  Screenshot   this one \nrest", HOOKS) == HOOKS[1]


def test_match_hook_no_match_and_bad_input():
    assert match_hook("something else entirely\nx", HOOKS) is None
    assert match_hook("", HOOKS) is None
    assert match_hook(None, HOOKS) is None


def test_set_hook_fills_only_missing(tmp_path):
    p = tmp_path / "ledger.jsonl"
    rows = [{"media_id": "111", "hook": None}, {"media_id": "222", "hook": "kept"}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    assert ledger.set_hook("111", "new", path=p) is True
    assert ledger.set_hook("222", "overwrite", path=p) is False
    got = {r["media_id"]: r["hook"] for r in ledger.load_entries(p)}
    assert got == {"111": "new", "222": "kept"}
