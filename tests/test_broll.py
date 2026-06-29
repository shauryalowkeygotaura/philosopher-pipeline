"""Offline unit tests for the B-roll modules.

These cover the pure logic only (query sanitization, theme derivation, numeric
clamping) so the suite stays network-free and deterministic. The actual fetch
and ffmpeg compose are exercised by manual smoke runs, not here.
"""

import math

import broll_fetcher as bf


# --- _safe_query: whitelist sanitization (project rule) --------------------
def test_safe_query_strips_lucene_operators():
    dirty = 'stoicism) OR mediatype:(audio) AND "x"'
    clean = bf._safe_query(dirty)
    assert all(c.isalnum() or c == " " for c in clean)
    assert ":" not in clean and "(" not in clean and '"' not in clean


def test_safe_query_collapses_whitespace_and_caps_length():
    assert bf._safe_query("  ocean   waves  ") == "ocean waves"
    assert len(bf._safe_query("word " * 200)) <= 120


def test_safe_query_handles_non_string():
    # Must not raise on non-string input (e.g. None from a failed fetch).
    assert isinstance(bf._safe_query(None), str)
    assert isinstance(bf._safe_query(12345), str)


# --- derive_broll_theme: quote -> atmospheric theme ------------------------
def test_derive_theme_maps_keywords():
    assert bf.derive_broll_theme("We suffer more in imagination") == "storm clouds sky"
    assert bf.derive_broll_theme("Time is a river of passing events") == "flowing river water"


def test_derive_theme_is_deterministic_fallback():
    q = "An entirely abstract proposition without imagery"
    assert bf.derive_broll_theme(q) == bf.derive_broll_theme(q)
    assert bf.derive_broll_theme(q) in bf._THEME_POOL


def test_derive_theme_handles_none_quote():
    # Must not raise on a None quote (fetch_quote failures, etc.)
    assert bf.derive_broll_theme(None, "Plato") in bf._THEME_POOL + [t for _, t in bf._THEME_MAP]


# --- compose numeric clamping (NaN/inf safety) -----------------------------
def test_compose_clamps_bad_numbers(monkeypatch):
    import broll_compose as bc

    captured = {}

    def fake_run(cmd):
        # capture the final ffmpeg command so we can assert no nan/inf leaked
        captured["cmd"] = cmd

    # Stub out all ffmpeg/PIL side effects; we only test the clamping path.
    monkeypatch.setattr(bc, "_run", fake_run)
    monkeypatch.setattr(bc, "_normalize_clip", lambda s, d: d.write_bytes(b"x"))
    monkeypatch.setattr(bc, "_build_text_overlay", lambda *a, **k: a[-1])

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "c.mp4"
        clip.write_bytes(b"x")
        out = Path(td) / "out.mp4"
        bc.compose_broll_reel(
            [clip], "q", "p", None, out, "fonts/PlayfairDisplay-Regular.ttf",
            reel_duration=float("nan"), music_volume=float("inf"),
        )
    joined = " ".join(captured["cmd"])
    assert "nan" not in joined.lower() and "inf" not in joined.lower()
