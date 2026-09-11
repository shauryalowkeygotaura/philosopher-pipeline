"""
tests/test_bandit_ledger.py -- self-improving reel loop (Phases 1-2).

Covers:
  * ledger media-id whitelist + record/load/attach round-trip
  * uploader records a real media id but NOT a mock (no test pollution)
  * bandit.pick_hook is byte-identical to the legacy round-robin until data
    accrues, then biases toward the best-performing arm
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import bandit  # noqa: E402
import ledger  # noqa: E402


HOOKS = [
    "hook-a", "hook-b", "hook-c", "hook-d",
    "hook-e", "hook-f", "hook-g", "hook-h",
]


# ── ledger: media-id whitelist ────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    ("123456789", "123456789"),
    ("123_456", "123_456"),
    ("  987 ", "987"),
    ("abc", None),
    ("12a", None),
    ("", None),
    (None, None),
    ("12-34", None),
])
def test_validate_media_id(value, expected):
    assert ledger.validate_media_id(value) == expected


def test_extract_media_id_prefers_pk():
    media = MagicMock()
    media.pk = "111"
    media.id = "111_222"
    assert ledger.extract_media_id(media) == "111"


def test_extract_media_id_from_dict_falls_back_to_id():
    assert ledger.extract_media_id({"pk": None, "id": "111_222"}) == "111_222"


def test_extract_media_id_rejects_bare_mock():
    """A MagicMock with no real id must yield None (so tests never pollute the ledger)."""
    assert ledger.extract_media_id(MagicMock()) is None


# ── ledger: record / load / attach round-trip ─────────────────────────────────

def test_record_and_load(tmp_path):
    led = tmp_path / "ledger.jsonl"
    mid = ledger.record_upload(
        "100200300", mp4_path="/x/plato-2026.mp4",
        philosopher="Plato", hook="hook-c", path=led,
    )
    assert mid == "100200300"
    rows = ledger.load_entries(led)
    assert len(rows) == 1
    assert rows[0]["media_id"] == "100200300"
    assert rows[0]["hook"] == "hook-c"
    assert rows[0]["mp4"] == "plato-2026.mp4"  # basename only
    assert rows[0]["insights"] is None


def test_record_skips_invalid_media(tmp_path):
    led = tmp_path / "ledger.jsonl"
    assert ledger.record_upload(MagicMock(), path=led) is None
    assert ledger.load_entries(led) == []


def test_attach_insights_round_trip(tmp_path):
    led = tmp_path / "ledger.jsonl"
    ledger.record_upload("555", hook="hook-a", path=led)
    assert ledger.attach_insights("555", {"saved": 9}, path=led) is True
    rows = ledger.load_entries(led)
    assert rows[0]["insights"] == {"saved": 9}
    assert ledger.pending_insight_ids(led) == []


def test_load_skips_malformed_lines(tmp_path):
    led = tmp_path / "ledger.jsonl"
    led.write_text('{"media_id": "1"}\nnot-json\n{"media_id": "2"}\n', encoding="utf-8")
    rows = ledger.load_entries(led)
    assert [r["media_id"] for r in rows] == ["1", "2"]


# ── bandit: byte-identical until data accrues ─────────────────────────────────

def test_pick_hook_matches_legacy_round_robin_with_no_data():
    for post_count in range(20):
        legacy = HOOKS[post_count % len(HOOKS)]
        assert bandit.pick_hook("Plato", post_count, HOOKS, entries=[]) == legacy


def test_reward_weighting_prefers_saves():
    assert bandit.reward({"saved": 1}) == 3.0
    assert bandit.reward({"like_count": 5}) == 5.0
    assert bandit.reward({"reach": 100}) == 100.0  # coarse fallback
    assert bandit.reward({}) is None
    assert bandit.reward(None) is None


def test_pick_exploits_best_arm_when_data_exists():
    entries = [
        {"hook": "hook-a", "insights": {"saved": 1}},      # reward 3
        {"hook": "hook-c", "insights": {"saved": 50}},     # reward 150 (winner)
        {"hook": "hook-d", "insights": {"like_count": 2}},  # reward 2
    ]
    # epsilon=0 -> pure exploit, must choose the highest-reward arm regardless
    # of what the round-robin baseline would have returned.
    for post_count in range(8):
        chosen = bandit.pick_hook("Plato", post_count, HOOKS, entries=entries, epsilon=0.0)
        assert chosen == "hook-c"


def test_pick_explores_to_baseline_under_high_epsilon():
    entries = [{"hook": "hook-c", "insights": {"saved": 50}}]
    # epsilon=1 -> always explore -> always the deterministic round-robin pick.
    for post_count in range(8):
        assert bandit.pick_hook("Plato", post_count, HOOKS, entries=entries, epsilon=1.0) == HOOKS[post_count % len(HOOKS)]


def test_live_insights_flag_default_off(monkeypatch):
    monkeypatch.delenv("PHILOSOPHER_LIVE_INSIGHTS", raising=False)
    assert bandit.live_insights_enabled() is False
    monkeypatch.setenv("PHILOSOPHER_LIVE_INSIGHTS", "1")
    assert bandit.live_insights_enabled() is True


# ── uploader integration: real id recorded, mock id not ───────────────────────

def test_upload_reel_records_real_media_id(tmp_path, monkeypatch):
    monkeypatch.setenv("INSTAGRAM_USERNAME", "u")
    monkeypatch.setenv("INSTAGRAM_PASSWORD", "p")
    import uploader
    uploader._client = None
    monkeypatch.setattr(uploader, "_PIPELINE_DIR", tmp_path)

    led = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(ledger, "LEDGER_PATH", led)

    mp4 = tmp_path / "plato.mp4"
    mp4.write_bytes(b"\x00" * 8)

    posted = MagicMock()
    posted.pk = "778899"
    mock_client = MagicMock()
    mock_client.clip_upload.return_value = posted

    with patch("instagrapi.Client", return_value=mock_client):
        result = uploader.upload_reel(
            str(mp4), "caption", meta={"philosopher": "Plato", "hook": "hook-c"}
        )

    assert result is True
    rows = ledger.load_entries(led)
    assert len(rows) == 1
    assert rows[0]["media_id"] == "778899"
    assert rows[0]["hook"] == "hook-c"
    uploader._client = None


# ---------------------------------------------------------------------------
# loop_status honesty tests
#
# 2026-09-10: loop_status() called without `arms` counted EVERY reward row in
# the ledger, including 134 backfilled rows that carry no hook, and reported
# "learning". The loop was in round-robin the whole time. A diagnostic that
# reports green on a dead loop is the reason nobody looked for a month, so
# these tests pin the failure modes it must name out loud.

def _row(hook=None, likes=0, comments=0, **extra):
    row = {"hook": hook, "insights": {"like_count": likes, "comment_count": comments}}
    row.update(extra)
    return row


def test_loop_status_does_not_call_orphaned_rewards_learning():
    """Rewards on rows with no arm value cannot drive selection."""
    import bandit
    rows = [_row(hook=None, likes=5) for _ in range(20)]
    st = bandit.loop_status(entries=rows)
    assert st["learning"] is False
    assert st["phase"] == 1
    assert st["reward_observations"] == 0      # attributable, not raw
    assert st["rewards_in_ledger"] == 20
    assert st["orphaned_rewards"] == 20
    assert "ORPHANED" in st["verdict"]


def test_loop_status_flags_a_single_observed_arm_as_degenerate():
    """One arm with data is not a comparison; exploit just replays it."""
    import bandit
    rows = [_row(hook="only hook", likes=3) for _ in range(6)]
    st = bandit.loop_status(entries=rows)
    assert st["arms_with_data"] == 1
    assert st["learning"] is False
    assert "DEGENERATE" in st["verdict"]


def test_loop_status_names_all_zero_insights_as_a_reach_problem():
    """Insights present and zero on every metric is not 'no data yet'."""
    import bandit
    rows = [_row(hook="a hook", likes=0, comments=0) for _ in range(35)]
    st = bandit.loop_status(entries=rows)
    assert st["zero_signal_rows"] == 35
    assert st["rewards_in_ledger"] == 0
    assert st["learning"] is False
    assert "DEAD SIGNAL" in st["verdict"]


def test_loop_status_reports_learning_only_with_two_live_arms():
    import bandit
    arms = ["a", "b"]
    rows = [_row(hook="a", likes=4), _row(hook="b", likes=9)]
    st = bandit.loop_status(arms, entries=rows)
    assert st["arms_with_data"] == 2
    assert st["learning"] is True
    assert st["phase"] == 2
    assert "LEARNING" in st["verdict"]


def test_loop_status_ignores_rewards_for_retired_arms():
    """A hook removed from the arm list stops counting, and says so."""
    import bandit
    rows = [_row(hook="retired wording", likes=7) for _ in range(9)]
    st = bandit.loop_status(["current a", "current b"], entries=rows)
    assert st["reward_observations"] == 0
    assert st["orphaned_rewards"] == 9
    assert st["learning"] is False


# ---------------------------------------------------------------------------
# close_the_loop: the wiring that was missing
#
# bandit.loop_status() and insights.refresh_pending() were both well written and
# had ZERO callers in the repo. Every pipeline run finished without ever asking
# whether the reel it just shipped taught the channel anything. These tests pin
# that the run now asks, and that asking can never sink an upload.

def test_close_the_loop_reports_status_without_network(monkeypatch):
    import pipeline, bandit
    monkeypatch.setattr(bandit, "live_insights_enabled", lambda: False)
    out = pipeline.close_the_loop(["hook a", "hook b"])
    assert out["refreshed"] == 0                 # flag off means no network
    assert out["status"] is not None
    assert "verdict" in out["status"]


def test_close_the_loop_survives_a_broken_insights_pull(monkeypatch):
    """A diagnostic must never be able to kill a run that already published."""
    import pipeline, insights
    monkeypatch.setattr(insights, "refresh_pending",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ig down")))
    out = pipeline.close_the_loop(["hook a"])
    assert out["refreshed"] == 0
    assert out["status"] is not None             # status still reported


def test_close_the_loop_survives_a_broken_status(monkeypatch):
    import pipeline, bandit
    monkeypatch.setattr(bandit, "loop_status",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad ledger")))
    out = pipeline.close_the_loop(["hook a"])
    assert out["status"] is None                 # degraded, not raised
