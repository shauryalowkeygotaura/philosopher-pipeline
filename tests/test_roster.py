"""
test_roster.py -- the promotion bench that keeps the pool unbounded.

Promotion writes to philosophers.md, which is the pipeline's source of truth
and is keyed to state.json history, so the invariants that matter are: never
promote a name off the bench, never duplicate an existing entry, never drop
an existing entry.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import roster
from input_parser import parse_philosophers


@pytest.fixture
def roster_file(tmp_path):
    p = tmp_path / "philosophers.md"
    p.write_text("# Philosophers\n\n- Albert Camus\n- Voltaire\n", encoding="utf-8")
    return p


# --- bench contents -------------------------------------------------------

def test_every_candidate_has_bio_and_vibes():
    for name, (bio, vibes) in roster.CANDIDATES.items():
        assert bio.strip(), f"{name} has no bio"
        assert vibes, f"{name} has no vibes"
        assert all(isinstance(v, str) and v for v in vibes)


def test_promoted_names_stay_in_the_catalog():
    """CANDIDATES is a catalog, not a queue.

    A promoted philosopher must REMAIN in CANDIDATES: it is the only source of
    their bio and song vibes, so removing them on promotion would ship captions
    with no bio and break song matching. Seneca was promoted 2026-08-27 and is
    both active and still catalogued.
    """
    active = parse_philosophers(Path(__file__).resolve().parent.parent / "philosophers.md")
    for name in active:
        if name in roster.CANDIDATES:
            assert roster.bio(name), f"{name} is active but has no bio"
            assert roster.vibes(name), f"{name} is active but has no vibes"


def test_available_never_offers_an_active_name():
    """The real dedup invariant: promotion must never be a no-op."""
    active = parse_philosophers(Path(__file__).resolve().parent.parent / "philosophers.md")
    offered = {n.lower() for n in roster.available(active)}
    for name in active:
        assert name.lower() not in offered, f"{name} is active but still offered"


def test_bio_and_vibes_lookup_is_case_insensitive():
    assert roster.bio("seneca") == roster.bio("Seneca") != ""
    assert roster.vibes("SENECA") == roster.vibes("Seneca")


def test_bio_and_vibes_unknown_name_is_empty():
    assert roster.bio("Nobody At All") == ""
    assert roster.vibes("Nobody At All") == []


# --- availability ---------------------------------------------------------

def test_available_excludes_active_names():
    out = roster.available(["Seneca", "Albert Camus"])
    assert "Seneca" not in out
    assert "Epictetus" in out


def test_available_is_case_insensitive():
    assert "Seneca" not in roster.available(["seneca"])


def test_available_preserves_promotion_order():
    out = roster.available([])
    assert out == list(roster.CANDIDATES)


# --- promotion ------------------------------------------------------------

def test_promote_appends_to_file(roster_file):
    assert roster.promote("Seneca", path=roster_file) is True
    names = parse_philosophers(roster_file)
    assert names == ["Albert Camus", "Voltaire", "Seneca"]


def test_promote_is_idempotent(roster_file):
    assert roster.promote("Seneca", path=roster_file) is True
    assert roster.promote("Seneca", path=roster_file) is False
    assert parse_philosophers(roster_file).count("Seneca") == 1


def test_promote_refuses_names_off_the_bench(roster_file):
    """Only vetted, public-domain thinkers may reach the account."""
    assert roster.promote("Some Living Influencer", path=roster_file) is False
    assert parse_philosophers(roster_file) == ["Albert Camus", "Voltaire"]


def test_promote_never_drops_existing_entries(roster_file):
    before = parse_philosophers(roster_file)
    roster.promote("Seneca", path=roster_file)
    roster.promote("Epictetus", path=roster_file)
    after = parse_philosophers(roster_file)
    assert after[:len(before)] == before


def test_promote_handles_file_without_trailing_newline(tmp_path):
    p = tmp_path / "philosophers.md"
    p.write_text("# Philosophers\n\n- Voltaire", encoding="utf-8")
    assert roster.promote("Seneca", path=p) is True
    assert parse_philosophers(p) == ["Voltaire", "Seneca"]


def test_promote_normalizes_case_to_canonical_spelling(roster_file):
    assert roster.promote("seneca", path=roster_file) is True
    assert "Seneca" in parse_philosophers(roster_file)


def test_promote_missing_file_returns_false(tmp_path):
    assert roster.promote("Seneca", path=tmp_path / "nope.md") is False


# --- integration with the caption/song layers -----------------------------

def test_promoted_philosopher_gets_a_bio_from_get_bio():
    """Without the roster fallback, promoted names ship a caption with no bio."""
    from fetcher import get_bio

    assert get_bio("Seneca").strip() != ""


def test_promoted_philosopher_gets_vibes_for_song_matching():
    from fetcher import match_song

    songs = [
        {"url": "https://youtube.com/watch?v=aaa", "label": "stoic resolute drums"},
        {"url": "https://youtube.com/watch?v=bbb", "label": "bubbly pop happy"},
    ]
    picked = match_song("Seneca", "q", songs, used_in_run=[], used_for_philosopher=[])
    assert picked == "https://youtube.com/watch?v=aaa"


def test_existing_philosophers_still_prefer_their_own_vibes():
    """The roster fallback must not shadow PHILOSOPHER_VIBES."""
    from fetcher import PHILOSOPHER_VIBES, get_bio

    assert PHILOSOPHER_VIBES["albert camus"]
    assert "absurdist" in get_bio("Albert Camus").lower()
