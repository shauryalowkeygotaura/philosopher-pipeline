"""
models.py -- one place that knows which Groq models exist.

WHY THIS EXISTS
---------------
On 2026-08-26 Groq removed the entire Llama line from this account.
`llama-3.3-70b-versatile` and `llama-3.1-8b-instant` both started returning
404 model_not_found, and because every call site caught the exception and fell
back to a hardcoded default, nothing broke loudly:

  - fetch_slogan() shipped its fallback string, so 13 of 23 recent reels
    published the identical punchline "Truth is found alone in the dark"
  - generate_quotes() returned [], so reels quietly degraded to LRU replay

Model ids were hardcoded in four separate call sites across two files, so the
outage had four independent places to hide. They now live here.

TWO TIERS, EACH A FALLBACK CHAIN
--------------------------------
A decommission should degrade to the next-best MODEL, never straight to a
hardcoded string. `chat()` walks the tier's chain and only gives up when every
model 404s, so the next removal costs a log line rather than three weeks of
identical slogans.

MULTIPLE KEYS = MULTIPLE QUOTAS
-------------------------------
Groq meters per ORGANISATION, not per key, so extra keys minted from the same
account share one 200k/day ceiling and buy nothing. Keys from a DIFFERENT Groq
account each carry their own quota, and those do add up. `chat()` rotates to
the next key when the current one hits its DAILY cap (never on a per-minute
throttle -- that just needs a short wait, and burning a fresh key on it wastes
the very capacity the extra key was for).

TOKEN FLOOR
-----------
The gpt-oss models emit reasoning tokens before their visible answer. A budget
that was ample for Llama truncates them into invalid JSON -- measured: the same
JSON-mode request fails at max_tokens=60 and succeeds at 400. Callers keep
passing the budget they need for the ANSWER; this module raises it to a floor
that leaves room to think.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Sequence

log = logging.getLogger(__name__)

# Env vars holding Groq keys, in the order they are tried. Each should come
# from a DIFFERENT Groq account; same-account keys share one quota.
KEY_VARS = ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3", "GROQ_API_KEY_4")

# Index of the key currently in use. Module-level so one exhausted key is not
# retried by every subsequent call in the same run.
_key_index = 0


def api_keys() -> list[str]:
    """Every configured key, in priority order, de-duplicated.

    Accepts either the numbered vars in KEY_VARS or a comma-separated
    GROQ_API_KEYS. Duplicates are dropped: the same key twice would look like
    added capacity and silently is not.
    """
    import os

    found: list[str] = []
    bulk = os.environ.get("GROQ_API_KEYS", "")
    for raw in bulk.split(","):
        k = raw.strip()
        if k and k not in found:
            found.append(k)
    for var in KEY_VARS:
        k = (os.environ.get(var) or "").strip()
        if k and k not in found:
            found.append(k)
    return found


def _make_client(key: str) -> Any:
    from groq import Groq
    return Groq(api_key=key)


def get_client() -> Any:
    """A Groq client on the current key. Raises if none are configured."""
    keys = api_keys()
    if not keys:
        raise RuntimeError("no Groq API key configured (set GROQ_API_KEY)")
    return _make_client(keys[min(_key_index, len(keys) - 1)])


def _rotate_key() -> bool:
    """Advance to the next configured key. False when none are left."""
    global _key_index
    keys = api_keys()
    if _key_index + 1 >= len(keys):
        return False
    _key_index += 1
    log.warning("models: daily quota spent; switching to key %d of %d.",
                _key_index + 1, len(keys))
    return True


def reset_keys() -> None:
    """Start again from the first key (new run, or tests)."""
    global _key_index
    _key_index = 0

# Quote recall and attribution verification, where a wrong answer publishes a
# fabricated quotation under a real person's name.
#
# Ordered by MEASUREMENT, not by size. Re-run scripts/eval_verifier.py after any
# model change; on 18 known-genuine and 8 known-fabricated quotes (2026-08-26):
#
#     openai/gpt-oss-20b     recall 39%   false-accept 0%   <- best
#     openai/gpt-oss-120b    recall 28%   false-accept 0%
#     qwen/qwen3.8-27b       recall 28%   false-accept 0%
#     llama-3.3-70b (gone)   recall 50%   false-accept 0%
#
# The bigger model is not the better verifier here. False-accept is the number
# that must stay 0; recall is throughput and is compensated by asking for more
# candidates (quotes.generate_quotes n=...).
SMART: tuple[str, ...] = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
)

# Short, high-volume, low-stakes generation (slogans, theme labels).
FAST: tuple[str, ...] = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
)

# Reasoning models spend tokens before answering. Below this they truncate
# mid-thought and JSON mode rejects the result.
MIN_MAX_TOKENS = 500

# Groq enforces BOTH a per-minute and a per-day token ceiling, and reports both
# as HTTP 429. They need opposite responses:
#
#   TPM (8,000/min on free)  -> transient. Wait the seconds Groq names, retry.
#   TPD (200,000/day)        -> terminal for today. Stop; retrying burns nothing
#                               but time and the quota is already gone.
#
# Treating a TPM throttle as terminal killed a 2026-08-27 maintenance run after
# 7 calls with "RATE LIMITED" when the retry window was 2.5 SECONDS.
MAX_THROTTLE_WAIT = 90.0   # seconds; longer than this is not worth blocking on
MAX_THROTTLE_RETRIES = 4


def parse_retry_after(exc: Exception) -> float | None:
    """Seconds Groq asks us to wait, from the 429 body. None if unstated."""
    m = re.search(r"try again in ([0-9.]+)\s*s", str(exc))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    m = re.search(r"try again in (?:(\d+)m)?([0-9.]+)s", str(exc))
    if m:
        return float(m.group(1) or 0) * 60 + float(m.group(2))
    return None


def is_daily_limit(exc: Exception) -> bool:
    """True for the per-DAY ceiling, which no amount of waiting fixes today."""
    text = str(exc)
    return "per day" in text or "(TPD)" in text or "TPD" in text


def is_throttle(exc: Exception) -> bool:
    """True for any 429, daily or per-minute."""
    text = str(exc)
    return "rate_limit_exceeded" in text or "Error code: 429" in text


def _extra_params(model: str) -> dict[str, Any]:
    """Per-family params that are not portable across the chain.

    gpt-oss reasons before answering; on a 4-7 word slogan that can burn the
    whole budget and return empty content. `reasoning_effort="low"` keeps the
    thinking short. Sending it to qwen would 400, so it is applied by family
    rather than passed by callers.
    """
    return {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}


def _is_missing_model(exc: Exception) -> bool:
    """True when the error means 'this model is gone', not 'your call was bad'."""
    text = str(exc)
    return "model_not_found" in text or "does not exist or you do not have access" in text


def chat(
    client: Any,
    tier: Sequence[str],
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    require_content: bool = False,
    **kwargs: Any,
) -> Any:
    """Run a chat completion against the first model in `tier` that exists.

    Raises the last exception if every model in the chain is gone, so the
    caller's own error handling still applies. Any non-404 error (rate limit,
    bad request, network) propagates immediately rather than silently retrying
    against a weaker model.

    `require_content=True` also advances the chain when a model returns an
    EMPTY message. A reasoning model that spends its whole budget thinking
    answers with 200 OK and no content, which is indistinguishable from a
    refusal at the call site and, for fetch_slogan, silently ships the
    hardcoded fallback line.
    """
    budget = max(max_tokens, MIN_MAX_TOKENS)
    last: Exception | None = None
    for model in tier:
      attempt = 0
      while True:
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=budget,
                **_extra_params(model), **kwargs
            )
            if require_content:
                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    log.warning(
                        "models.chat: %s returned empty content (likely spent the "
                        "budget reasoning); trying the next in the chain.", model)
                    # break, NOT continue: `continue` belongs to the retry loop
                    # and would re-ask the same model forever.
                    break
            return resp
        except Exception as e:  # noqa: BLE001 - inspected, then re-raised
            if _is_missing_model(e):
                log.warning("models.chat: %s is gone; trying the next in the chain.", model)
                last = e
                break  # next model in the chain
            # A spent DAILY quota is what spare keys exist for. Rotate and
            # retry the same model; only give up when every key is spent.
            if is_throttle(e) and is_daily_limit(e) and _rotate_key():
                client = get_client()
                continue
            # A per-minute throttle is not a failure, it is backpressure.
            if (is_throttle(e) and not is_daily_limit(e)
                    and attempt < MAX_THROTTLE_RETRIES):
                wait = parse_retry_after(e)
                if wait is not None and wait <= MAX_THROTTLE_WAIT:
                    attempt += 1
                    log.info("models.chat: throttled, waiting %.1fs (attempt %d/%d)",
                             wait + 0.5, attempt, MAX_THROTTLE_RETRIES)
                    time.sleep(wait + 0.5)
                    continue
            raise
    log.error("models.chat: every model in %s is unavailable. Update models.py.", list(tier))
    raise last if last is not None else RuntimeError("no models configured")


def available(client: Any) -> list[str]:
    """Model ids this account can currently reach. Diagnostic helper."""
    try:
        return sorted(m.id for m in client.models.list().data)
    except Exception as e:  # noqa: BLE001
        log.warning("models.available: %s", e)
        return []


def check(client: Any) -> dict[str, list[str]]:
    """Report which configured models are live vs gone.

    Run this after any Groq outage: `doppler run -- python -c
    "import models,groq;print(models.check(groq.Groq()))"`
    """
    live = set(available(client))
    if not live:
        return {}
    out: dict[str, list[str]] = {}
    for name, tier in (("SMART", SMART), ("FAST", FAST)):
        out[name] = [m for m in tier if m in live]
        gone = [m for m in tier if m not in live]
        if gone:
            log.warning("%s tier has dead entries: %s", name, gone)
    return out
