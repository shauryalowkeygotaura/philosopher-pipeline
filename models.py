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
from typing import Any, Sequence

log = logging.getLogger(__name__)

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
                    continue
            return resp
        except Exception as e:  # noqa: BLE001 - inspected, then re-raised
            if not _is_missing_model(e):
                raise
            log.warning("models.chat: %s is gone; trying the next in the chain.", model)
            last = e
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
