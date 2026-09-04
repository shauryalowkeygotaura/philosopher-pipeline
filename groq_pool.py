# ---------------------------------------------------------------------------
# VENDORED FILE - DO NOT EDIT HERE.
# Canonical copy: Vault/Scripts/groq_pool.py
# Regenerate with: python Scripts/sync_groq_pool.py
# Local edits are overwritten on the next sync.
# ---------------------------------------------------------------------------
"""
groq_pool.py -- one Groq client layer, shared by every project in the vault.

CANONICAL COPY. Edit this file, then run Scripts/sync_groq_pool.py to push it
into each project. Do not edit the copies: they are overwritten.

WHAT IT SOLVES
--------------
Three failures hit this vault in one week, each of which was invisible because
the call site caught the error and fell back to a hardcoded string:

1. Groq removed the Llama line (2026-08-26). Model ids were hardcoded at four
   call sites in philosopher-pipeline and two in autoshop, so the outage had
   six places to hide. One reel series published the SAME slogan 13 times out
   of 23 before anyone noticed.
2. A per-minute throttle was mistaken for a dead daily quota, killing a
   maintenance run after 7 of 60 calls over a 2.5-SECOND wait.
3. A single account's 200k/day ceiling became the hard limit on how fast work
   could run, while unused keys from other accounts sat in other projects.

FOUR AXES OF FALLBACK
---------------------
    key    -> rotate on a spent DAILY quota, or a dead/revoked key (401)
    model  -> walk the tier chain on 404 model_not_found
    time   -> wait out a per-minute throttle, which Groq sizes for you
    budget -> raise every max_tokens to a floor so reasoning models can think

KEYS ADD UP ONLY ACROSS ACCOUNTS
--------------------------------
Groq meters per ORGANISATION. Two keys from one account share one 200k/day and
buy nothing; keys from separate accounts each carry their own. Verified
2026-08-27 by spending tokens on one key and watching the other's counter stay
independent. Duplicates are dropped here so the same key listed twice cannot
masquerade as extra capacity.

USAGE
-----
    import groq_pool

    resp = groq_pool.chat(
        groq_pool.SMART,                       # or FAST, or your own tuple
        messages=[{"role": "user", "content": "..."}],
        max_tokens=800,
    )
    text = resp.choices[0].message.content

Per-project model preferences: set groq_pool.SMART / FAST at import time, or
pass an explicit tuple as the first argument.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Sequence

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------

# Checked in order. Each should be from a DIFFERENT Groq account; same-account
# keys share one quota. GROQ_API_KEYS (comma-separated) is also accepted.
KEY_VARS = ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4")

_key_index = 0


# A value that is ENTIRELY template words and separators, e.g.
# "paste_your_key_here", "<YOUR_API_KEY>", "changeme", "xxx".
#
# Anchored with fullmatch on purpose. An unanchored search for "xxx" or
# "example" would fire on those letters occurring by chance inside a real
# credential's random body, and a key that vanishes silently is far worse to
# debug than one that earns an honest 401. A real key contains segments that
# are not in this word list, so it cannot fullmatch.
_PLACEHOLDER_WORD = r"paste|your|api|key|token|secret|here|placeholder|changeme|change|replace|me|dummy|example|sample|todo|fixme|none|null|value|goes|insert|add|x+"
_PLACEHOLDER_KEY = re.compile(
    r"(?i)^[<\[{(]?(?:%s)(?:[_\-. ]?(?:%s))*[>\]})]?$"
    % (_PLACEHOLDER_WORD, _PLACEHOLDER_WORD)
)


def _usable_key(value: str, source: str) -> bool:
    """Drop template values; keep anything that might be a real credential.

    autoshop has carried the literal `paste_your_key_here` in GROQ_API_KEY
    since at least 2026-08. Nothing broke loudly, because the pool rotates past
    a dead key - it just spent the first call of every run earning a 401 and
    made every log open on a failure.

    Deliberately NOT "must start with gsk_". That rule is tempting and wrong:
    it would also discard a key routed through a Groq-compatible gateway. An
    unrecognised value is therefore kept and flagged, and only a value that is
    wholly a template is refused.
    """
    if value.startswith("gsk_"):
        return True
    masked = (value[:4] + "...") if len(value) > 4 else "..."
    if _PLACEHOLDER_KEY.fullmatch(value):
        log.warning("groq_pool: ignoring %s=%r - it is a placeholder, not a key.",
                    source, masked)
        return False
    log.warning("groq_pool: %s=%r does not start with gsk_. Using it anyway; "
                "if it 401s, that is why.", source, masked)
    return True


def api_keys() -> list[str]:
    """Every configured key, in priority order, de-duplicated and stripped."""
    found: list[str] = []
    for raw in (os.environ.get("GROQ_API_KEYS") or "").split(","):
        k = raw.strip()
        if k and k not in found and _usable_key(k, "GROQ_API_KEYS"):
            found.append(k)
    for var in KEY_VARS:
        k = (os.environ.get(var) or "").strip()
        if k and k not in found and _usable_key(k, var):
            found.append(k)
    return found


def key_count() -> int:
    return len(api_keys())


def reset_keys() -> None:
    """Start again at the first key. Call once at the top of a long run."""
    global _key_index
    _key_index = 0


def _make_client(key: str) -> Any:
    from groq import Groq
    return Groq(api_key=key)


def get_client() -> Any:
    """A client on the currently-active key."""
    keys = api_keys()
    if not keys:
        raise RuntimeError(
            "No Groq API key configured. Set GROQ_API_KEY (and GROQ_API_KEY_2/_3 "
            "from OTHER Groq accounts to multiply the daily quota)."
        )
    return _make_client(keys[min(_key_index, len(keys) - 1)])


def _rotate_key(reason: str = "unusable") -> bool:
    """Move to the next key. False when every key is spent."""
    global _key_index
    keys = api_keys()
    if _key_index + 1 >= len(keys):
        return False
    _key_index += 1
    log.warning("groq_pool: key %d %s; switching to key %d of %d.",
                _key_index, reason, _key_index + 1, len(keys))
    return True


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

# Defaults as of 2026-08-27. Override per project by reassigning these.
#
# Ordered by MEASUREMENT where it has been measured. For philosopher-pipeline's
# quote-attribution gate, gpt-oss-20b beat gpt-oss-120b on recall at equal
# (zero) false-accept: the bigger model is not automatically the better one.
SMART: tuple[str, ...] = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
)

FAST: tuple[str, ...] = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
)

# Reasoning models spend tokens before answering; below this they truncate
# mid-thought and JSON mode rejects the result. Measured: the same JSON-mode
# request fails at max_tokens=60 and succeeds at 400.
MIN_MAX_TOKENS = 500

# Head-room subtracted on top of the server-reported overshoot when shrinking a
# rejected request. Groq's own token count differs slightly from the one it
# quotes back, and a retry that misses by three tokens costs another round trip.
TOKEN_SHRINK_MARGIN = 256

MAX_THROTTLE_WAIT = 90.0    # seconds; longer is not worth blocking a job on
MAX_THROTTLE_RETRIES = 4


def _extra_params(model: str) -> dict[str, Any]:
    """Params that are not portable across model families."""
    # gpt-oss reasons before answering. On a short answer that can consume the
    # whole budget and return empty content. Sending this to qwen would 400.
    return {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}


# --------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------

def is_missing_model(exc: Exception) -> bool:
    """The model is gone, as opposed to the request being wrong."""
    t = str(exc)
    return "model_not_found" in t or "does not exist or you do not have access" in t


def is_bad_key(exc: Exception) -> bool:
    """The key itself is rejected: revoked, mistyped, or a placeholder.

    autoshop shipped `gsk_..._here` in Doppler and every call 401'd. A dead key
    is not a dead service, so the right move is the same as an exhausted one:
    move to the next key rather than fail the run.
    """
    t = str(exc)
    return "invalid_api_key" in t or "Invalid API Key" in t or "Error code: 401" in t


def is_throttle(exc: Exception) -> bool:
    """Any 429, per-minute or per-day.

    Checked AFTER is_request_too_large: a 413 body also carries
    `rate_limit_exceeded`, and treating it as backpressure means waiting for a
    minute that will never make the request fit.
    """
    t = str(exc)
    return "rate_limit_exceeded" in t or "Error code: 429" in t


def is_request_too_large(exc: Exception) -> bool:
    """The single request exceeds the per-minute token budget.

    Groq counts prompt + max_tokens against TPM, so reserving a large output is
    itself the thing that breaks. autoshop asked for max_tokens=8192 against a
    TPM of 8000 and got a 413 on every attempt for two straight weeks.

    Nothing about this is transient. Waiting does not help, because the request
    is larger than a whole minute of budget. Rotating keys does not help,
    because the limit is per-organisation. Only a smaller request helps.
    """
    t = str(exc)
    return ("Request too large" in t
            or "Error code: 413" in t
            or "reduce your message size" in t)


def parse_token_budget(exc: Exception) -> tuple[int, int] | None:
    """(limit, requested) out of 'Limit 8000, Requested 8612'.

    Using the server's own arithmetic beats guessing a safe ceiling: the limit
    differs per model and per tier, and it changes without notice.
    """
    m = re.search(r"Limit (\d+), Requested (\d+)", str(exc))
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except ValueError:
        return None


def is_daily_limit(exc: Exception) -> bool:
    """The per-DAY ceiling, which waiting will not clear soon."""
    t = str(exc)
    return "per day" in t or "TPD" in t


def parse_retry_after(exc: Exception) -> float | None:
    """Seconds Groq asks us to wait. Handles both '2.5s' and '3m39.02s'."""
    m = re.search(r"try again in (?:(\d+)m)?([0-9.]+)s", str(exc))
    if m:
        try:
            return float(m.group(1) or 0) * 60 + float(m.group(2))
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------
# The call
# --------------------------------------------------------------------------

def chat(
    tier: Sequence[str] = SMART,
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    require_content: bool = False,
    client: Any = None,
    **kwargs: Any,
) -> Any:
    """Chat completion with key, model, time and budget fallback.

    `require_content=True` also advances the model chain when a model answers
    200 OK with an EMPTY message. A reasoning model that spends its whole
    budget thinking does exactly that, and at the call site it is
    indistinguishable from a refusal -- which is how one project silently
    shipped its hardcoded fallback line on 13 of 23 outputs.

    Raises the last error when every model and key is exhausted, so the
    caller's own error handling still applies.
    """
    budget = max(max_tokens, MIN_MAX_TOKENS)
    # Key rotation only applies to clients we build. A caller-supplied client
    # is bound to its own key; silently swapping it would ignore what they
    # passed and could send their request to a different account.
    owns_client = client is None
    active = client or get_client()
    last: Exception | None = None

    for model in tier:
        attempt = 0
        while True:
            try:
                resp = active.chat.completions.create(
                    model=model, messages=messages, max_tokens=budget,
                    **_extra_params(model), **kwargs
                )
                if require_content:
                    content = (resp.choices[0].message.content or "").strip()
                    if not content:
                        log.warning("groq_pool: %s returned empty content; "
                                    "trying the next model.", model)
                        last = RuntimeError(
                            f"{model} returned empty content (likely spent its "
                            f"token budget reasoning; raise max_tokens)")
                        break  # next model, NOT a retry of this one
                return resp
            except Exception as e:  # noqa: BLE001 - classified, then re-raised
                if is_missing_model(e):
                    log.warning("groq_pool: %s is gone; trying the next model.", model)
                    last = e
                    break
                # Checked BEFORE the throttle branch: a 413 body contains
                # `rate_limit_exceeded` too, and reading it as backpressure
                # means sleeping through a minute that cannot help.
                if is_request_too_large(e):
                    pair = parse_token_budget(e)
                    shrunk = False
                    if pair and budget > MIN_MAX_TOKENS:
                        limit, requested = pair
                        # Give back exactly the overshoot, plus a margin, so
                        # the retry lands under the limit on the first try.
                        target = max(MIN_MAX_TOKENS,
                                     budget - (requested - limit) - TOKEN_SHRINK_MARGIN)
                        if target < budget:
                            log.warning(
                                "groq_pool: %s rejected %d tokens against a "
                                "limit of %d; retrying with max_tokens=%d.",
                                model, requested, limit, target)
                            budget = target
                            shrunk = True
                    if shrunk:
                        continue
                    # Either the prompt ALONE is over the limit, or the budget
                    # is already at the floor. Another model may have a bigger
                    # allowance; if none does, the caller gets this error.
                    log.warning("groq_pool: %s cannot fit this request even at "
                                "max_tokens=%d; trying the next model.",
                                model, budget)
                    last = e
                    break
                # A spent DAILY quota, or a key that is simply dead, is
                # exactly what the spare keys are for.
                if is_bad_key(e) or (is_throttle(e) and is_daily_limit(e)):
                    reason = ("was rejected (401 invalid key)" if is_bad_key(e)
                              else "has spent its daily quota")
                    if owns_client and _rotate_key(reason):
                        active = get_client()
                        continue
                # A per-minute throttle is backpressure, not failure. Never
                # rotate a key for it: that burns the capacity the spare key
                # was added to provide, over a wait of a couple of seconds.
                if (is_throttle(e) and not is_daily_limit(e)
                        and attempt < MAX_THROTTLE_RETRIES):
                    wait = parse_retry_after(e)
                    if wait is not None and wait <= MAX_THROTTLE_WAIT:
                        attempt += 1
                        log.info("groq_pool: throttled, waiting %.1fs (%d/%d)",
                                 wait + 0.5, attempt, MAX_THROTTLE_RETRIES)
                        time.sleep(wait + 0.5)
                        continue
                raise

    log.error("groq_pool: every model in %s is unavailable.", list(tier))
    raise last if last is not None else RuntimeError("no models configured")


def health() -> dict[str, Any]:
    """Which keys and models are actually usable right now.

    Run after any Groq outage:
        doppler run -- python -c "import groq_pool,json; print(json.dumps(groq_pool.health(),indent=2))"
    """
    keys = api_keys()
    out: dict[str, Any] = {"keys_configured": len(keys), "keys": [], "models": []}
    live_models: set[str] = set()
    for i, key in enumerate(keys, 1):
        entry = {"index": i, "suffix": key[-4:], "live": False, "error": None}
        try:
            client = _make_client(key)
            ids = sorted(m.id for m in client.models.list().data)
            entry["live"] = True
            live_models |= set(ids)
        except Exception as e:  # noqa: BLE001
            entry["error"] = str(e)[:100]
        out["keys"].append(entry)
    checked = bool(live_models)
    out["models_checked"] = checked
    for name, tier in (("SMART", SMART), ("FAST", FAST)):
        out["models"].append({
            "tier": name,
            # None, not [], when no key could list models: "nothing missing"
            # and "could not look" must not read the same.
            "available": [m for m in tier if m in live_models] if checked else None,
            "gone": [m for m in tier if m not in live_models] if checked else None,
        })
    return out
