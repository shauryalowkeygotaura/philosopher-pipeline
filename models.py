"""
models.py -- this project's Groq model preferences.

The client machinery (key rotation, model fallback, throttle waits, token
floor) lives in the vendored groq_pool.py, shared by every project in the
vault. This file exists only to hold the model ORDERING specific to
philosopher-pipeline, and to keep `import models` working at the call sites.

WHY THE ORDER HERE DIFFERS FROM THE SHARED DEFAULT
--------------------------------------------------
The attribution gate in quotes.verify_quotes is the only thing standing
between Groq and a fabricated quotation published under a real person's name,
and its behaviour is a property of the MODEL, not the code. Measured on 18
known-genuine and 8 known-fabricated quotes (2026-08-26):

    openai/gpt-oss-20b     recall 39%   false-accept 0%   <- leads SMART here
    openai/gpt-oss-120b    recall 28%   false-accept 0%
    qwen/qwen3.8-27b       recall 28%   false-accept 0%
    llama-3.3-70b (gone)   recall 50%   false-accept 0%

The bigger model is the worse verifier for this task, which is why SMART is
ordered by measurement rather than size, and why the shared default (which
leads with 120b) is overridden here. Re-run scripts/eval_verifier.py after ANY
model change: false-accept must stay 0, recall is only throughput.
"""
from __future__ import annotations

import groq_pool

# Re-exported so existing call sites keep working unchanged.
from groq_pool import (  # noqa: F401
    KEY_VARS,
    MAX_THROTTLE_RETRIES,
    MAX_THROTTLE_WAIT,
    MIN_MAX_TOKENS,
    api_keys,
    chat,
    get_client,
    health,
    is_daily_limit,
    is_missing_model,
    is_throttle,
    key_count,
    parse_retry_after,
    reset_keys,
)

# Quote recall and attribution verification. Ordered by measurement; see above.
SMART: tuple[str, ...] = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
)

# Slogans and theme labels: short, high volume, low stakes.
FAST: tuple[str, ...] = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
)

# groq_pool.chat() falls back to its own module-level defaults when no tier is
# passed, so point those at this project's ordering too.
groq_pool.SMART = SMART
groq_pool.FAST = FAST


def check(client=None) -> dict:
    """Backwards-compatible alias for groq_pool.health()."""
    return groq_pool.health()
