"""
test_models.py -- model chain, throttle handling, and multi-key rotation.

Targets groq_pool, the vendored module shared by every project in the vault.
models.py is now only this project's tier ordering.

Groq meters per ORGANISATION, so extra keys only add capacity when they come
from a different Groq account. The rotation exists for that case: when one
account's DAILY quota is spent, move to the next account's key.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import groq_pool as models

TPM = ("Error code: 429 - {'error': {'message': 'Rate limit reached on tokens per "
       "minute (TPM): Limit 8000, Used 7291. Please try again in 2.5125s.', "
       "'code': 'rate_limit_exceeded'}}")
TPD = ("Error code: 429 - {'error': {'message': 'Rate limit reached on tokens per "
       "day (TPD): Limit 200000, Used 199618. Please try again in 11m3.12s.', "
       "'code': 'rate_limit_exceeded'}}")
GONE = "Error code: 404 - {'error': {'code': 'model_not_found'}}"


class Err(Exception):
    def __init__(self, msg): self.msg = msg
    def __str__(self): return self.msg


@pytest.fixture(autouse=True)
def clean_keys(monkeypatch):
    for var in models.KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    models.reset_keys()
    yield
    models.reset_keys()


class FakeClient:
    """Records calls and raises a scripted sequence of errors."""

    def __init__(self, errors=None, content="ok"):
        self.errors = list(errors or [])
        self.content = content
        self.calls = []
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kw):
        self.calls.append(kw)
        if self.errors:
            raise Err(self.errors.pop(0))
        msg = type("M", (), {"content": self.content})()
        return type("R", (), {"choices": [type("Ch", (), {"message": msg})()]})()


# --- key discovery --------------------------------------------------------

def test_no_keys_configured(monkeypatch):
    assert models.api_keys() == []
    with pytest.raises(RuntimeError):
        models.get_client()


def test_numbered_vars_are_collected_in_order(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "one")
    monkeypatch.setenv("GROQ_API_KEY_2", "two")
    assert models.api_keys() == ["one", "two"]


def test_comma_separated_var_is_supported(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "a, b ,c")
    assert models.api_keys() == ["a", "b", "c"]


def test_duplicate_keys_are_dropped(monkeypatch):
    """The same key twice looks like added capacity and is not."""
    monkeypatch.setenv("GROQ_API_KEY", "same")
    monkeypatch.setenv("GROQ_API_KEY_2", "same")
    assert models.api_keys() == ["same"]


def test_blank_values_ignored(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "real")
    monkeypatch.setenv("GROQ_API_KEY_2", "   ")
    assert models.api_keys() == ["real"]


# --- rotation -------------------------------------------------------------

def test_rotate_advances_then_runs_out(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "a")
    monkeypatch.setenv("GROQ_API_KEY_2", "b")
    assert models._rotate_key() is True
    assert models._rotate_key() is False


def test_daily_limit_rotates_to_the_next_key(monkeypatch):
    """The whole point of a second account's key.

    No client is passed: rotation only applies to clients groq_pool builds
    itself, since a caller-supplied one is bound to its own key.
    """
    monkeypatch.setenv("GROQ_API_KEY", "spent")
    monkeypatch.setenv("GROQ_API_KEY_2", "fresh")
    built = []

    def fake_make(key):
        built.append(key)
        # Only the first key is out of daily quota.
        return FakeClient(errors=[TPD] if key == "spent" else [])

    monkeypatch.setattr(models, "_make_client", fake_make)
    resp = models.chat(("m",), messages=[], max_tokens=10)
    assert resp is not None
    assert built == ["spent", "fresh"], f"expected rotation, built {built}"


def test_caller_supplied_client_is_never_swapped(monkeypatch):
    """Rotating a client the caller handed us would ignore their key choice."""
    monkeypatch.setenv("GROQ_API_KEY", "spent")
    monkeypatch.setenv("GROQ_API_KEY_2", "fresh")
    built = []
    monkeypatch.setattr(models, "_make_client",
                        lambda k: (built.append(k), FakeClient())[1])
    with pytest.raises(Err):
        models.chat(("m",), messages=[], max_tokens=10,
                    client=FakeClient(errors=[TPD]))
    assert built == [], "must not build a client on another key"


def test_daily_limit_with_no_spare_key_propagates(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "only")
    monkeypatch.setattr(models, "_make_client", lambda k: FakeClient(errors=[TPD]))
    with pytest.raises(Err):
        models.chat(("m",), messages=[], max_tokens=10)


def test_per_minute_throttle_does_not_burn_a_spare_key(monkeypatch):
    """A 2.5s wait must not consume the fresh account's quota."""
    monkeypatch.setenv("GROQ_API_KEY", "a")
    monkeypatch.setenv("GROQ_API_KEY_2", "b")
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    rotated = []
    monkeypatch.setattr(models, "_make_client",
                        lambda k: (rotated.append(k), FakeClient())[1])
    client = FakeClient(errors=[TPM])
    models.chat(("m",), messages=[], max_tokens=10, client=client)
    assert rotated == [], "per-minute throttle must not rotate keys"
    assert models._key_index == 0


# --- model chain ----------------------------------------------------------

def test_missing_model_advances_the_chain(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "a")
    client = FakeClient(errors=[GONE])
    models.chat(("dead", "alive"), messages=[], max_tokens=10, client=client)
    assert [c["model"] for c in client.calls] == ["dead", "alive"]


def test_token_budget_is_floored(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "a")
    client = FakeClient()
    models.chat(("m",), messages=[], max_tokens=24, client=client)
    assert client.calls[0]["max_tokens"] >= models.MIN_MAX_TOKENS


def test_reasoning_effort_only_for_gpt_oss():
    assert models._extra_params("openai/gpt-oss-20b") == {"reasoning_effort": "low"}
    assert models._extra_params("qwen/qwen3.8-27b") == {}


def test_require_content_advances_on_empty_answer(monkeypatch):
    """An exhausted reasoning budget answers 200 OK with nothing."""
    monkeypatch.setenv("GROQ_API_KEY", "a")
    client = FakeClient(content="   ")
    with pytest.raises(Exception):
        models.chat(("m",), messages=[], max_tokens=10, require_content=True, client=client)


def test_non_throttle_errors_propagate_immediately(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "a")
    client = FakeClient(errors=["connection reset by peer"])
    with pytest.raises(Err):
        models.chat(("m",), messages=[], max_tokens=10, client=client)


BAD_KEY = "Error code: 401 - {'error': {'message': 'Invalid API Key', 'code': 'invalid_api_key'}}"


def test_dead_key_rotates_to_the_next(monkeypatch):
    """autoshop shipped a placeholder key and every call 401'd.

    A dead key is not a dead service; move on rather than fail the run.
    """
    monkeypatch.setenv("GROQ_API_KEY", "placeholder")
    monkeypatch.setenv("GROQ_API_KEY_2", "real")
    built = []

    def fake_make(key):
        built.append(key)
        return FakeClient(errors=[BAD_KEY] if key == "placeholder" else [])

    monkeypatch.setattr(models, "_make_client", fake_make)
    assert models.chat(("m",), messages=[], max_tokens=10) is not None
    assert built == ["placeholder", "real"]


def test_bad_key_classification():
    assert models.is_bad_key(Err(BAD_KEY))
    assert not models.is_bad_key(Err(TPD))
    assert not models.is_bad_key(Err(GONE))


def test_all_keys_dead_propagates(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "a")
    monkeypatch.setenv("GROQ_API_KEY_2", "b")
    monkeypatch.setattr(models, "_make_client", lambda k: FakeClient(errors=[BAD_KEY]))
    with pytest.raises(Err):
        models.chat(("m",), messages=[], max_tokens=10)
