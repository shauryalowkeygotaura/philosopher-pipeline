"""
eval_verifier.py -- measure the attribution gate before trusting it.

WHY
---
verify_quotes() is the only thing standing between Groq and a fabricated
quotation published under a real person's name. Its behaviour is a property of
the MODEL, not of the code, so it must be re-measured whenever the model
changes. On 2026-08-26 Groq removed the Llama line and the replacement scored
materially differently:

    qwen/qwen3.8-27b (gone)  recall 50%   false-accept 0%
    openai/gpt-oss-20b              recall 39%   false-accept 0%
    openai/gpt-oss-120b             recall 28%   false-accept 0%
    qwen/qwen3.8-27b                recall 28%   false-accept 0%

THE TWO NUMBERS ARE NOT EQUAL IN IMPORTANCE
-------------------------------------------
false-accept MUST be 0. A single leak publishes an invented line attributed to
a real thinker. recall is only throughput: a rejected genuine quote costs one
more API call, and is compensated by raising `n` in generate_quotes.

So: pick the model with the best recall AMONG those at 0% false-accept. Never
trade a leak for yield.

USAGE
-----
    doppler run -- python scripts/eval_verifier.py
    doppler run -- python scripts/eval_verifier.py --model openai/gpt-oss-120b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import models  # noqa: E402
import quotes  # noqa: E402
from fetcher import PHILOSOPHER_QUOTES  # noqa: E402

# GENUINE: the curated static pool is hand-verified real.
GENUINE = [
    (name.title(), q)
    for name, qs in PHILOSOPHER_QUOTES.items()
    for q in qs[:3]
    if name != "søren kierkegaard"  # duplicate key of the ascii spelling
][:18]

# FABRICATED: plausible inventions, near-paraphrases of real passages, and
# truncations. Every one of these has actually been emitted by a generator.
FABRICATED = [
    ("Albert Camus", "Defiance is the only way"),
    ("Albert Camus", "The absurd man says yes and his effort is unending forever."),
    ("Franz Kafka", "Solitude is the price of mastery and the wage of genius."),
    ("Voltaire", "Those who can make you believe absurdities"),
    ("Friedrich Nietzsche", "You must have chaos within you"),
    ("Marcus Aurelius", "Discipline is the bridge between goals and accomplishment."),
    ("Immanuel Kant", "Reason is the compass that never rusts in the storm."),
    ("Blaise Pascal", "The heart computes what the mind cannot measure."),
]


def evaluate(label: str) -> tuple[float, float]:
    passed = sum(quotes.verify_quotes(p, [q])[0] for p, q in GENUINE)
    leaks = [(p, q) for p, q in FABRICATED if quotes.verify_quotes(p, [q])[0]]
    recall = passed / len(GENUINE)
    false_accept = len(leaks) / len(FABRICATED)

    print(f"\n{label}")
    print(f"  genuine passed : {passed}/{len(GENUINE)}  (recall {recall:.0%})")
    print(f"  fakes   passed : {len(leaks)}/{len(FABRICATED)}  "
          f"(false-accept {false_accept:.0%})  <- MUST be 0")
    for p, q in leaks:
        print(f"    LEAK: {p} | {q}")
    if leaks:
        print("\n  FAIL: this configuration would publish a fabricated quotation.")
    return recall, false_accept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="evaluate one model instead of the SMART tier")
    ap.add_argument("--all", action="store_true",
                    help="evaluate every model in the SMART tier separately")
    args = ap.parse_args()

    if args.model:
        models.SMART = (args.model,)
        _, fa = evaluate(args.model)
        return 1 if fa else 0

    if args.all:
        original = models.SMART
        worst = 0.0
        for m in original:
            models.SMART = (m,)
            _, fa = evaluate(m)
            worst = max(worst, fa)
        models.SMART = original
        return 1 if worst else 0

    _, fa = evaluate(f"SMART tier (leads with {models.SMART[0]})")
    return 1 if fa else 0


if __name__ == "__main__":
    raise SystemExit(main())
