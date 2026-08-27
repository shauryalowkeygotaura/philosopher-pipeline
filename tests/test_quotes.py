"""
test_quotes.py -- regression + unit tests for the reward-steered quote supply.

The headline test is test_exhausted_pool_never_repeats_consecutively: the old
fetch_quote() returned PHILOSOPHER_QUOTES[0] on every call once a philosopher
was spent, which published the same quote 30+ times. Any change that lets two
consecutive selections collide must fail here.

Every test runs with allow_generation=False and a tmp pool path so the suite
makes no network call and never touches runs/quote_pool.json.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import quotes
from fetcher import PHILOSOPHER_QUOTES, fetch_quote


@pytest.fixture
def pool_path(tmp_path):
    return tmp_path / "quote_pool.json"


# --- canonical form -------------------------------------------------------

def test_canon_collapses_dash_variants():
    a = "Don't walk in front of me — I may not follow."
    b = "Don't walk in front of me - I may not follow"
    assert quotes.canon(a) == quotes.canon(b)


def test_canon_collapses_curly_apostrophes_and_case():
    assert quotes.canon("Man's Fate") == quotes.canon("man’s fate")


def test_canon_collapses_internal_punctuation():
    a = "In the midst of winter, I found there was, within me, an invincible summer."
    b = "In the midst of winter I found there was within me an invincible summer"
    assert quotes.canon(a) == quotes.canon(b)


def test_canon_keeps_distinct_quotes_distinct():
    a = "In the midst of winter, I found an invincible summer."
    b = "In the depth of winter, I finally learned there lay an invincible summer."
    assert quotes.canon(a) != quotes.canon(b)


def test_canon_handles_non_string():
    assert quotes.canon(None) == ""
    assert quotes.canon(42) == ""


# --- acceptance policy ----------------------------------------------------

def test_is_acceptable_rejects_too_short_and_too_long():
    assert not quotes.is_acceptable("Too short.")
    assert not quotes.is_acceptable("x" * (quotes.MAX_QUOTE_CHARS + 1))


def test_is_acceptable_rejects_llm_preamble():
    assert not quotes.is_acceptable('Here is a quote about the human condition and fate.')
    assert not quotes.is_acceptable('As an AI, I can offer this reflection on mortality.')


def test_is_acceptable_rejects_baked_in_attribution():
    assert not quotes.is_acceptable("Man is condemned to be free, and must choose - Sartre")


def test_is_acceptable_rejects_near_duplicate_of_avoid_set():
    existing = {quotes.canon("I rebel; therefore I exist.")}
    assert not quotes.is_acceptable("I rebel, therefore I exist", avoid=existing)


def test_is_acceptable_accepts_a_clean_quote():
    assert quotes.is_acceptable("The only way to deal with an unfree world is to become absolutely free.")


def test_is_acceptable_rejects_truncation_of_known_quote():
    """Regression: 2026-08-01 Groq returned a pool quote with its ending cut off.

    Exact-key dedup is blind to this; only containment catches it.
    """
    full = ("The only way to deal with an unfree world is to become so absolutely "
            "free that your very existence is an act of rebellion.")
    truncated = "The only way to deal with an unfree world is to become so absolutely free"
    assert not quotes.is_acceptable(truncated, avoid={quotes.canon(full)})


def test_is_acceptable_rejects_expansion_of_known_quote():
    short = "Common sense is not so common."
    expanded = "Common sense is not so common, and rarer still among the powerful."
    assert not quotes.is_acceptable(expanded, avoid={quotes.canon(short)})


def test_is_acceptable_rejects_short_generic_invention():
    """"Defiance is the only way" (24 chars) was a real fabrication Groq emitted."""
    assert not quotes.is_acceptable("Defiance is the only way")


def test_is_acceptable_rejects_paraphrase_of_known_quote():
    """Regression: Groq emitted two rewordings of the dancing-star line.

    Neither canon() nor containment can see this -- only token overlap can.
    """
    a = "You must have chaos within you to give birth to a dancing star."
    b = "You need chaos in your soul to give birth to a dancing star."
    assert not quotes.is_acceptable(b, avoid={quotes.canon(a)})


def test_similarity_separates_paraphrase_from_distinct_quotes():
    """The 0.6 threshold must sit in the gap between these two populations."""
    a = quotes.content_tokens(quotes.canon(
        "Those who can make you believe absurdities can make you commit atrocities."))
    b = quotes.content_tokens(quotes.canon(
        "Those who make you believe absurdities are also capable of making you commit atrocities."))
    assert quotes.similarity(a, b) > quotes.MAX_TOKEN_OVERLAP

    # Every genuinely distinct pair among one thinker's real quotes must clear it.
    real = PHILOSOPHER_QUOTES["friedrich nietzsche"]
    for i in range(len(real)):
        for j in range(i + 1, len(real)):
            s = quotes.similarity(
                quotes.content_tokens(quotes.canon(real[i])),
                quotes.content_tokens(quotes.canon(real[j])),
            )
            assert s <= quotes.MAX_TOKEN_OVERLAP, f"{real[i]!r} vs {real[j]!r} = {s}"


def test_similarity_handles_empty_token_sets():
    assert quotes.similarity(frozenset(), frozenset({"a"})) == 0.0


def test_containment_guard_ignores_very_short_overlaps():
    """Two distinct short quotes sharing an opening must not collide."""
    assert quotes.is_acceptable(
        "The unexamined life is not worth living at all.",
        avoid={quotes.canon("The unexamined")},
    )


# --- attribution verification --------------------------------------------

def test_verify_quotes_fails_closed_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert quotes.verify_quotes("Camus", ["anything at all here"]) == [False]


def test_verify_quotes_empty_input():
    assert quotes.verify_quotes("Camus", []) == []


def test_verify_quotes_raises_when_not_fail_closed(monkeypatch):
    """Destructive callers must be able to tell 'rejected' from 'check failed'.

    Regression: prune() deletes on a negative verdict. With fail_closed=True a
    missing key or a 429 reads as "every quote is fabricated" and would wipe
    runs/quote_pool.json.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(quotes.VerificationUnavailable):
        quotes.verify_quotes("Camus", ["a candidate line long enough to pass"],
                             fail_closed=False)


def test_verify_quotes_propagates_network_failure_when_not_fail_closed(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("429 rate limited")

    monkeypatch.setitem(sys.modules, "groq", type("M", (), {"Groq": Boom}))
    with pytest.raises(quotes.VerificationUnavailable):
        quotes.verify_quotes("Camus", ["a candidate line long enough to pass"],
                             fail_closed=False)


def test_verify_quotes_swallows_network_failure_when_fail_closed(monkeypatch):
    """The publishing path must never raise -- it degrades to rejecting all."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("429 rate limited")

    monkeypatch.setitem(sys.modules, "groq", type("M", (), {"Groq": Boom}))
    assert quotes.verify_quotes("Camus", ["a candidate line long enough"]) == [False]


def test_generate_quotes_returns_empty_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert quotes.generate_quotes("Camus", theme="defiance") == []


GENUINE = "A genuine documented line of real philosophy here."
INVENTED = "An invented line that the checker will refuse to bless."


def test_generate_quotes_drops_unverified(monkeypatch):
    """Anything the attribution check rejects must never reach the pool."""
    monkeypatch.setattr(quotes, "request_candidates",
                        lambda *a, **k: [GENUINE, INVENTED])
    monkeypatch.setattr(quotes, "verify_quotes", lambda p, c: [True, False])
    out = quotes.generate_quotes("Camus", theme="defiance")
    assert [r["quote"] for r in out] == [GENUINE]
    assert out[0]["theme"] == "defiance"
    assert out[0]["source"] == "groq"


def test_generate_quotes_filters_before_verifying(monkeypatch):
    """Unacceptable candidates are dropped without spending a verify call."""
    seen = {}

    def fake_verify(philosopher, candidates):
        seen["candidates"] = list(candidates)
        return [True] * len(candidates)

    monkeypatch.setattr(quotes, "request_candidates",
                        lambda *a, **k: ["short", GENUINE, "Here is a fine quotation for you."])
    monkeypatch.setattr(quotes, "verify_quotes", fake_verify)
    out = quotes.generate_quotes("Camus", theme="defiance")
    assert seen["candidates"] == [GENUINE]  # "short" + preamble already gone
    assert len(out) == 1


def test_generate_quotes_respects_avoid_set(monkeypatch):
    monkeypatch.setattr(quotes, "request_candidates", lambda *a, **k: [GENUINE])
    monkeypatch.setattr(quotes, "verify_quotes", lambda p, c: [True] * len(c))
    out = quotes.generate_quotes("Camus", theme="defiance",
                                 avoid={quotes.canon(GENUINE)})
    assert out == []


def test_generate_quotes_survives_candidate_failure(monkeypatch):
    monkeypatch.setattr(quotes, "request_candidates", lambda *a, **k: [])
    assert quotes.generate_quotes("Camus", theme="defiance") == []


def test_select_quote_falls_back_to_lru_when_generation_raises(monkeypatch, pool_path):
    """The LRU floor must survive an exception from the generation path."""
    def boom(*a, **k):
        raise RuntimeError("groq exploded")

    monkeypatch.setattr(quotes, "generate_quotes", boom)
    used = list(PHILOSOPHER_QUOTES["voltaire"])
    r = quotes.select_quote("Voltaire", used, allow_generation=True, pool_path=pool_path)
    assert r["reframed"] is True
    assert quotes.canon(r["quote"]) == quotes.canon(PHILOSOPHER_QUOTES["voltaire"][0])


# --- pool persistence -----------------------------------------------------

def test_add_to_pool_persists_and_dedupes(pool_path):
    rows = [{"quote": "A sufficiently long and perfectly valid quote here.", "theme": "time"}]
    assert quotes.add_to_pool("Voltaire", rows, path=pool_path) == 1
    # Same quote with different punctuation must not create a second entry.
    dupe = [{"quote": "A sufficiently long, and perfectly valid quote here", "theme": "time"}]
    assert quotes.add_to_pool("Voltaire", dupe, path=pool_path) == 0
    assert len(quotes.load_pool(pool_path)["voltaire"]) == 1


def test_known_elsewhere_excludes_own_quotes(pool_path):
    own = quotes.known_elsewhere("Voltaire", pool_path)
    for q in PHILOSOPHER_QUOTES["voltaire"]:
        assert quotes.canon(q) not in own


def test_known_elsewhere_includes_other_philosophers(pool_path):
    own = quotes.known_elsewhere("Voltaire", pool_path)
    assert quotes.canon(PHILOSOPHER_QUOTES["albert camus"][0]) in own


def test_known_elsewhere_covers_generated_pool(pool_path):
    line = "A distinctive generated line belonging to Kafka alone."
    quotes.add_to_pool("Franz Kafka", [{"quote": line, "theme": "time"}], path=pool_path)
    assert quotes.canon(line) in quotes.known_elsewhere("Voltaire", pool_path)
    assert quotes.canon(line) not in quotes.known_elsewhere("Franz Kafka", pool_path)


def test_cross_attribution_is_rejected(pool_path):
    """Regression: Pascal was handed Voltaire's absurdities/atrocities line."""
    voltaire_line = PHILOSOPHER_QUOTES["voltaire"][0]
    avoid = quotes.known_elsewhere("Blaise Pascal", pool_path)
    assert not quotes.is_acceptable(voltaire_line, avoid=avoid)


def test_load_pool_survives_corrupt_file(pool_path):
    pool_path.write_text("{not json", encoding="utf-8")
    assert quotes.load_pool(pool_path) == {}


def test_load_pool_drops_malformed_rows(pool_path):
    pool_path.write_text(
        json.dumps({"voltaire": [{"quote": "ok quote that is long enough"}, {"bad": 1}, "junk"]}),
        encoding="utf-8",
    )
    assert len(quotes.load_pool(pool_path)["voltaire"]) == 1


# --- theme classification -------------------------------------------------

def test_keyword_theme_always_returns_valid_arm():
    for quote in ("death comes for all", "zzz nothing matches here", ""):
        assert quotes._keyword_theme(quote) in quotes.THEMES


def test_classify_theme_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert quotes.classify_theme("All men must face death.") in quotes.THEMES


# --- theme bandit ---------------------------------------------------------

def test_pick_theme_round_robins_without_reward_data():
    """Phase-1 contract: no insights in the ledger -> plain round robin."""
    picked = [
        quotes.pick_theme("Voltaire", i, entries=[])
        for i in range(len(quotes.THEMES))
    ]
    assert picked == list(quotes.THEMES)


def test_pick_theme_exploits_highest_reward_theme():
    entries = [
        {"theme": "mortality", "insights": {"like_count": 500}},
        {"theme": "mortality", "insights": {"like_count": 400}},
        {"theme": "time", "insights": {"like_count": 5}},
    ]
    # epsilon=0 disables exploration so the exploit branch is deterministic.
    assert quotes.pick_theme("Voltaire", 3, entries=entries, epsilon=0.0) == "mortality"


def test_theme_stats_ignores_rows_without_insights():
    entries = [
        {"theme": "time", "insights": None},
        {"theme": "time", "insights": {"like_count": 10}},
    ]
    assert quotes.theme_stats(entries)["time"]["n"] == 1.0


def test_top_exemplars_ranks_by_reward_and_prefers_philosopher():
    entries = [
        {"philosopher": "Voltaire", "quote": "own low", "insights": {"like_count": 1}},
        {"philosopher": "Voltaire", "quote": "own high", "insights": {"like_count": 900}},
        {"philosopher": "Kafka", "quote": "other huge", "insights": {"like_count": 5000}},
    ]
    out = quotes.top_exemplars("Voltaire", entries=entries, limit=3)
    assert out[0] == "own high"      # own philosopher first
    assert out[1] == "own low"       # ...even below a bigger foreign score
    assert out[2] == "other huge"


def test_top_exemplars_empty_without_reward_data():
    assert quotes.top_exemplars("Voltaire", entries=[{"quote": "q", "insights": None}]) == []


# --- selection: the regression that matters -------------------------------

def test_fresh_quote_is_not_reframed(pool_path):
    r = quotes.select_quote("Voltaire", [], allow_generation=False, pool_path=pool_path)
    assert r["reframed"] is False
    assert r["theme"] in quotes.THEMES


def test_select_quote_never_returns_a_used_quote_while_fresh_exist(pool_path):
    used = []
    pool = list(PHILOSOPHER_QUOTES["voltaire"])
    for _ in range(len(pool)):
        r = quotes.select_quote(
            "Voltaire", used, post_count=len(used),
            allow_generation=False, pool_path=pool_path,
        )
        assert r["reframed"] is False, "still had fresh quotes but replayed one"
        assert quotes.canon(r["quote"]) not in quotes.canon_set(used)
        used.append(r["quote"])
    assert len(quotes.canon_set(used)) == len(pool)


def test_exhausted_pool_never_repeats_consecutively(pool_path):
    """THE regression test for the frozen-[0] bug.

    With the pool fully spent, the old code returned all_quotes[0] every single
    time. LRU replay must instead cycle, so no two consecutive selections match
    and every pool quote is revisited before any is seen twice.
    """
    pool = list(PHILOSOPHER_QUOTES["voltaire"])
    used = list(pool)
    seen = []
    for i in range(len(pool) * 2):
        r = quotes.select_quote(
            "Voltaire", used, post_count=len(used),
            allow_generation=False, pool_path=pool_path,
        )
        assert r["reframed"] is True, "exhausted pool should mark replays"
        if seen:
            assert quotes.canon(r["quote"]) != quotes.canon(seen[-1]), (
                f"repeated {r['quote']!r} back to back on iteration {i}"
            )
        seen.append(r["quote"])
        used.append(r["quote"])
    # A full cycle of the pool must appear before anything repeats.
    assert len(quotes.canon_set(seen[: len(pool)])) == len(pool)


def test_lru_replay_picks_least_recently_used(pool_path):
    pool = list(PHILOSOPHER_QUOTES["voltaire"])
    # Oldest use = pool[0] (index 0); it must be the one replayed.
    used = list(pool)
    r = quotes.select_quote(
        "Voltaire", used, allow_generation=False, pool_path=pool_path,
    )
    assert quotes.canon(r["quote"]) == quotes.canon(pool[0])


def test_exhaustion_is_logged_as_error(pool_path, caplog):
    """Silent degradation is what let this run for 50+ reels. It must be loud."""
    used = list(PHILOSOPHER_QUOTES["voltaire"])
    with caplog.at_level("ERROR", logger="quotes"):
        quotes.select_quote(
            "Voltaire", used, allow_generation=False, pool_path=pool_path,
        )
    assert any("EXHAUSTED" in rec.message for rec in caplog.records)


def test_generated_pool_extends_available_quotes(pool_path):
    """A quote added to the persisted pool becomes selectable without Groq."""
    used = list(PHILOSOPHER_QUOTES["voltaire"])
    fresh = "Doubt is not a pleasant condition, but certainty is an absurd one."
    quotes.add_to_pool(
        "Voltaire", [{"quote": fresh, "theme": "absurdity"}], path=pool_path,
    )
    r = quotes.select_quote(
        "Voltaire", used, allow_generation=False, pool_path=pool_path,
    )
    assert quotes.canon(r["quote"]) == quotes.canon(fresh)
    assert r["reframed"] is False
    assert r["source"] == "groq"


def test_unknown_philosopher_returns_safe_fallback(pool_path):
    r = quotes.select_quote(
        "Nobody At All", [], allow_generation=False, pool_path=pool_path,
    )
    assert r["source"] == "fallback"
    assert len(r["quote"]) > 0


def test_dedup_is_canonical_not_exact(pool_path):
    """A punctuation variant of a used quote must not be served as fresh."""
    original = PHILOSOPHER_QUOTES["voltaire"][0]
    variant = original.replace(",", "").replace(".", "").replace("—", "-")
    r = quotes.select_quote(
        "Voltaire", [variant], allow_generation=False, pool_path=pool_path,
    )
    assert quotes.canon(r["quote"]) != quotes.canon(original)


# --- adapter back-compat --------------------------------------------------

def test_fetch_quote_still_returns_quote_and_reframed(pool_path):
    r = fetch_quote("Voltaire", used_quotes=[], allow_generation=False, pool_path=pool_path)
    assert isinstance(r["quote"], str) and r["quote"]
    assert r["reframed"] is False


def test_fetch_quote_passes_kwargs_through(pool_path):
    used = list(PHILOSOPHER_QUOTES["voltaire"])
    r = fetch_quote("Voltaire", used, allow_generation=False, pool_path=pool_path)
    assert r["reframed"] is True


# --- rate limiting --------------------------------------------------------

class _RateLimit(Exception):
    """Mimics the Groq SDK's 429 message shape."""
    def __str__(self):
        return ("Error code: 429 - {'error': {'message': 'Rate limit reached "
                "on tokens per day (TPD): Limit 200000', "
                "'code': 'rate_limit_exceeded'}}")


def test_is_rate_limit_detects_429():
    assert quotes._is_rate_limit(_RateLimit())
    assert not quotes._is_rate_limit(RuntimeError("connection reset"))
    assert not quotes._is_rate_limit(RuntimeError("model_not_found"))


def test_request_candidates_raises_on_rate_limit(monkeypatch):
    """A quota refusal must NOT look like 'the model returned nothing'.

    Returning [] would let the caller move to the next theme and spend more
    calls against a quota that is already gone.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class Boom:
        def __init__(self, *a, **k):
            raise _RateLimit()

    monkeypatch.setitem(sys.modules, "groq", type("M", (), {"Groq": Boom}))
    with pytest.raises(quotes.RateLimited):
        quotes.request_candidates("Camus", theme="defiance")


def test_verify_quotes_raises_on_rate_limit(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class Boom:
        def __init__(self, *a, **k):
            raise _RateLimit()

    monkeypatch.setitem(sys.modules, "groq", type("M", (), {"Groq": Boom}))
    with pytest.raises(quotes.RateLimited):
        quotes.verify_quotes("Camus", ["a candidate line long enough to count"])


def test_select_quote_survives_rate_limit(pool_path):
    """A reel must still ship when the quota is gone: LRU floor applies."""
    import unittest.mock as mock

    used = list(PHILOSOPHER_QUOTES["voltaire"])
    with mock.patch.object(quotes, "generate_quotes", side_effect=quotes.RateLimited("429")):
        r = quotes.select_quote("Voltaire", used, allow_generation=True, pool_path=pool_path)
    assert r["reframed"] is True
    assert r["quote"]


def test_rate_limited_is_not_verification_unavailable():
    """Distinct types: one means 'no quota', the other 'check could not run'."""
    assert not issubclass(quotes.RateLimited, quotes.VerificationUnavailable)
    assert not issubclass(quotes.VerificationUnavailable, quotes.RateLimited)


# --- per-minute vs per-day throttles --------------------------------------

_TPM = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model "
        "`openai/gpt-oss-20b` on tokens per minute (TPM): Limit 8000, Used 7291, "
        "Requested 1044. Please try again in 2.5125s.', 'code': 'rate_limit_exceeded'}}")
_TPD = ("Error code: 429 - {'error': {'message': 'Rate limit reached on tokens per "
        "day (TPD): Limit 200000, Used 199425. Please try again in 3m39.024s.', "
        "'code': 'rate_limit_exceeded'}}")


class _Throttle(Exception):
    def __init__(self, msg): self.msg = msg
    def __str__(self): return self.msg


def test_per_minute_throttle_is_not_terminal():
    """Regression: a 2.5-second throttle killed a whole maintenance run.

    Only the per-DAY ceiling justifies stopping. Treating the per-minute limit
    the same way aborted the 2026-08-27 sweep after 7 of 60 budgeted calls.
    """
    assert not quotes._is_rate_limit(_Throttle(_TPM))


def test_per_day_limit_is_terminal():
    assert quotes._is_rate_limit(_Throttle(_TPD))


def test_models_classifies_both_as_throttles():
    import models
    assert models.is_throttle(_Throttle(_TPM))
    assert models.is_throttle(_Throttle(_TPD))
    assert not models.is_daily_limit(_Throttle(_TPM))
    assert models.is_daily_limit(_Throttle(_TPD))


def test_retry_after_is_parsed_from_both_shapes():
    import models
    assert abs(models.parse_retry_after(_Throttle(_TPM)) - 2.5125) < 0.01
    # 3m39.024s -> 219.024
    assert abs(models.parse_retry_after(_Throttle(_TPD)) - 219.024) < 0.1


def test_non_throttle_errors_are_not_rate_limits():
    assert not quotes._is_rate_limit(_Throttle("model_not_found"))
    assert not quotes._is_rate_limit(_Throttle("connection reset by peer"))
