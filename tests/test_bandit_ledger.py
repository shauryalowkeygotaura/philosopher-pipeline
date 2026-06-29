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
