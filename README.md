# Philosopher Instagram Pipeline

## Quote supply and the self-improving loop

Quotes come from `quotes.py`, which walks a four-step supply chain and falls
through on failure:

1. a fresh quote matching the theme the bandit wants
2. any fresh quote, if that theme is out of stock
3. Groq generation for that theme, few-shot conditioned on the account's
   highest-engagement quotes, then persisted to `runs/quote_pool.json`
4. **LRU replay** — the quote used longest ago

Step 4 is the floor. Never remove it: it is what turns a missing `GROQ_API_KEY`
or a network failure into "maximally spaced repeat" instead of a frozen loop.

### The learning unit is the THEME, not the quote

Dedup guarantees each quote publishes once, so a per-quote bandit arm has `n=1`
forever: one noisy sample, never repeated, and "winning" would mean replaying a
quote dedup exists to prevent. Instead every quote carries a theme from a small
fixed vocabulary (`quotes.THEMES`). Themes recur across hundreds of posts,
accumulate real sample counts, and become the bandit arms via the existing
`bandit.arm_stats(arm_field="theme")`. Quotes stay disposable; taste accumulates.

### Attribution safety

Generated quotes are checked in three independent ways before they can ship,
because publishing a fabricated line under a real person's name is the failure
mode that matters:

- **`is_acceptable()`** — length, preamble/cliche markers, canonical-form
  dedup, containment (catches truncations), token-overlap (catches
  paraphrases), and `known_elsewhere()` (catches cross-attribution).
- **`verify_quotes()`** — a second zero-temperature call that must affirm the
  line is both genuinely documented AND complete. It **fails closed** for
  publishing, and raises `VerificationUnavailable` for destructive callers so a
  transient 429 is never mistaken for "everything is fake".
- **the LRU floor** — so rejecting everything still ships a reel.

Measured against 18 known-genuine and 8 known-fabricated quotes:

| configuration | recall | false-accept |
|---|---|---|
| strict prompt, temp 0 (**shipped**) | 50% | **0%** |
| + YES few-shot examples | 61% | 12% |
| + 3-vote unanimity | 61% | 12% |

Recall is traded away deliberately. The hardest case is a near-paraphrase of a
real passage ("The absurd man says yes and his effort is unending forever."
under Camus, where the real line ends "...his effort will henceforth be
unceasing") — the model believes that one *stably*, so self-consistency voting
costs 3x and buys nothing. **Raise yield by asking for more candidates
(`n=12`), never by loosening the gate.**

### Maintenance

```bash
# keep a buffer of verified quotes ahead of demand
doppler run -- python scripts/topup_pool.py --target 8

# re-verify pooled quotes after tightening the verifier (backs up first)
doppler run -- python scripts/topup_pool.py --prune --dry-run

# seed/refresh the bandit from reels already on Instagram
doppler run -- python scripts/backfill_ledger.py --dry-run
```

### Postmortem: the frozen-quote bug (2026-08-06)

`fetch_quote()` served `PHILOSOPHER_QUOTES[0]` forever once a philosopher's 6-7
quotes were spent. All 12 were exhausted, so **85 of 158 reels — 54% of the
account — republished a duplicate quote.** Each philosopher published its whole
pool, then repeated its first quote 7-8 more times. Three compounding causes,
all now fixed:

- exhaustion was a silent `[0]` fallback, not an error — it now logs `ERROR`
- dedup compared raw strings, so punctuation variants shrank the real pool
- `runs/upload_ledger.jsonl` was **not** in the CI `git add`, so every
  engagement row died with its ephemeral runner. Both bandits sat at their
  round-robin baseline for 158 posts with `insights=None`.

Anything the loop earns must be committed back by CI or it does not exist.

One debugging trap worth recording: the first pass at this was diagnosed
against a local clone **49 commits behind origin**, which understated the
damage (109 posts, not 158) and produced a phantom "CI stopped committing on
2026-06-27" bug that did not exist. `git fetch` before trusting `state.json`
or `runs/` — CI rewrites both on every cron slot.

## Reel styles (`STYLE` env var)

- `capcut` (default) — 7s fast-cut, beat-synced slideshow of paintings/portraits.
- `kinetic` — 28s letterbox + word-by-word typography (locked 5-beat format).
