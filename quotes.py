"""
quotes.py -- reward-steered quote supply for the philosopher pipeline.

WHY THIS EXISTS
---------------
fetcher.fetch_quote() used to serve from a static 79-quote table and, once a
philosopher's 6-7 quotes were spent, returned all_quotes[0] forever. By
2026-07-31 every one of the 12 philosophers was exhausted, so every reel
republished the same frozen first quote. This module replaces that dead end.

THE LEARNING UNIT IS THE THEME, NOT THE QUOTE
---------------------------------------------
Dedup guarantees each quote publishes exactly once, so a per-quote bandit arm
has n=1 forever: one noisy engagement sample, never repeated, and "winning"
would mean replaying a quote the dedup exists to prevent. Ranking on that is
overfitting to noise, not learning.

So every quote carries a THEME drawn from a small fixed vocabulary. Themes
recur across hundreds of posts, accumulate real sample counts, and become the
bandit arms (reusing bandit.arm_stats/pick, which are already generic over
arm_field). High-reward themes then steer Groq generation, few-shot conditioned
on the actual top-performing quotes. Individual quotes stay disposable; the
taste is what accumulates.

SUPPLY CHAIN (each step falls through to the next)
--------------------------------------------------
  1. fresh static/generated quote matching the bandit's chosen theme
  2. any fresh quote regardless of theme
  3. Groq generation for the chosen theme, persisted to the pool
  4. LRU replay -- the quote used longest ago

Step 4 exists so a missing GROQ_API_KEY or a network failure degrades to
"maximally spaced repeat" instead of resurrecting the frozen-[0] bug. It is the
floor of the system and must never be removed.

PERSISTENCE
-----------
The generated pool lives in runs/ (tracked by git and committed back by CI),
NOT cache/ (gitignored). The ledger was silently losing every row to exactly
this mistake -- see the 2026-07-31 postmortem in README.
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import bandit
import models

log = logging.getLogger(__name__)

_DIR = Path(__file__).parent.resolve()

# runs/ is tracked; cache/ is gitignored. Generated quotes are earned state and
# must survive the ephemeral CI runner, so they live here.
QUOTE_POOL_PATH = _DIR / "runs" / "quote_pool.json"

# The bandit arm vocabulary. Deliberately small and stable: arms only become
# statistically meaningful by recurring, so resist growing this list. Adding a
# theme resets that arm's sample count to zero.
THEMES: tuple[str, ...] = (
    "defiance",
    "mortality",
    "solitude",
    "absurdity",
    "self-mastery",
    "desire",
    "time",
    "suffering",
)

# Keyword fallback for classify_theme() when Groq is unavailable. Order matters:
# first theme with a hit wins, so the more distinctive markers come first.
_THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "mortality": ("death", "die", "dying", "dead", "mortal", "grave", "perish"),
    "absurdity": ("absurd", "meaning", "meaningless", "sisyphus", "purpose", "why"),
    "defiance": ("rebel", "revolt", "refuse", "defy", "resist", "free", "freedom", "chains"),
    "solitude": ("alone", "lonely", "solitude", "silence", "himself", "oneself"),
    "self-mastery": ("master", "discipline", "control", "habit", "virtue", "will", "become"),
    "desire": ("desire", "want", "wish", "passion", "love", "hunger", "crave"),
    "time": ("time", "moment", "present", "future", "past", "hour", "day", "eternity"),
    "suffering": ("suffer", "pain", "despair", "anguish", "misery", "wound", "burden"),
}

_DEFAULT_THEME = "absurdity"

# Quotes longer than this blow out the kinetic renderer's line breaking.
MAX_QUOTE_CHARS = 280
# Raised from 20 to 26 after a 2026-08-01 live test: Groq returned the invented
# aphorism "Defiance is the only way" (24 chars). Short generic lines are where
# fabrication concentrates -- a real documented quote carries more specificity.
# The shortest genuine quote in PHILOSOPHER_QUOTES is 26 chars, so this is the
# tightest bound that keeps every known-good line.
MIN_QUOTE_CHARS = 26

# A generated line must not be a truncation or expansion of one already held:
# the same live test returned "The only way to deal with an unfree world is to
# become so absolutely free", which is a pool quote with its ending lopped off.
# Exact-key comparison cannot see that; containment can.
MIN_CONTAINMENT_WORDS = 5

# Paraphrase guard. Containment only catches one quote sitting literally inside
# another; it is blind to reworded variants of the same line, which Groq emits
# readily -- a 2026-08-01 top-up produced both "You must have chaos within you
# to give birth to a dancing star." and "You need chaos in your soul to give
# birth to a dancing star." Jaccard overlap on content words catches those.
# 0.6 was chosen against the real pool: it separates the paraphrase pairs above
# (0.64-0.78) from genuinely distinct quotes by the same thinker (<0.4).
MAX_TOKEN_OVERLAP = 0.6

# Function words carry no signal about which quote this is, and leaving them in
# inflates overlap between any two English sentences.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on to for with
from by as at is are was were be been being am do does did have has had it its
i you he she we they me him her us them my your his our their not no nor so
one must can will would should could may might there here what which who whom
""".split())


# ---------------------------------------------------------------------------
# Canonical form (normalized dedup)
# ---------------------------------------------------------------------------

# Unicode dash and apostrophe variants that made "walk in front of me - I may
# not follow" and "walk in front of me, I may not lead" read as distinct quotes,
# silently shrinking the effective pool below its nominal 6-7 per philosopher.
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_APOSTROPHES = dict.fromkeys(map(ord, "‘’ʼ`´"), "'")
_QUOTEMARKS = dict.fromkeys(map(ord, "“”„«»"), '"')


def canon(quote: str) -> str:
    """Collapse a quote to a comparison key.

    Case, surrounding quotation marks, dash/apostrophe variants, internal
    punctuation and whitespace runs are all normalized away, so near-identical
    reposts collide instead of both entering the pool. Used for EVERY dedup
    decision in this module; never compare raw strings.
    """
    if not isinstance(quote, str):
        return ""
    text = unicodedata.normalize("NFKC", quote)
    text = text.translate(_DASHES).translate(_APOSTROPHES).translate(_QUOTEMARKS)
    text = text.lower().strip().strip('"').strip("'")
    # Drop everything that is not a letter, digit or space: punctuation variants
    # ("winter, I found" vs "winter I found") must not create a second identity.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def canon_set(quotes: Iterable[str]) -> set[str]:
    return {c for c in (canon(q) for q in quotes) if c}


def content_tokens(canonical: str) -> frozenset[str]:
    """Content words of an already-canonical string, for similarity scoring."""
    return frozenset(w for w in canonical.split() if w not in _STOPWORDS)


def similarity(a_tokens: frozenset[str], b_tokens: frozenset[str]) -> float:
    """Jaccard overlap of content words. 1.0 = identical content vocabulary."""
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


# ---------------------------------------------------------------------------
# Acceptance policy
# ---------------------------------------------------------------------------

# Phrases that mark an LLM ignoring the "output the quote only" instruction, or
# drifting into self-help cliche that does not match the account's voice.
_REJECT_MARKERS = (
    "as an ai", "here is", "here's a", "sure,", "certainly", "i hope",
    "follow your heart", "live your truth", "embrace the journey",
    "trust the process", "be yourself", "find your why",
)


def is_acceptable(quote: str, *, avoid: set[str] | None = None) -> bool:
    """Gate every generated quote before it can enter the pool.

    This is the single point where fabrication risk is managed, and it is the
    knob most worth tuning as you see what Groq actually returns. Current
    policy, deliberately conservative:
      - length within the renderer's limits
      - no preamble/refusal markers, no self-help cliche
      - not a near-duplicate of anything already in the pool (canonical form)
      - no attribution text baked into the quote body ("- Camus"), which the
        caption builder adds separately
    """
    if not isinstance(quote, str):
        return False
    text = quote.strip()
    if not (MIN_QUOTE_CHARS <= len(text) <= MAX_QUOTE_CHARS):
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in _REJECT_MARKERS):
        return False
    # An em-dash/hyphen attribution tail means the model appended the author.
    if re.search(r"[-–—]\s*\w+\s*$", text) and len(text.split()) > 4:
        return False
    if text.count("\n") > 1:
        return False
    key = canon(text)
    if not key:
        return False
    if not avoid:
        return True
    if key in avoid:
        return False
    # Containment: catch truncations/expansions of a quote we already hold,
    # which exact-key comparison is blind to. Guarded by a word floor so two
    # genuinely different short quotes cannot collide on a common opening.
    if len(key.split()) >= MIN_CONTAINMENT_WORDS:
        key_tokens = content_tokens(key)
        for known in avoid:
            if len(known.split()) < MIN_CONTAINMENT_WORDS:
                continue
            if key in known or known in key:
                return False
            # Paraphrase: same line reworded, which containment cannot see.
            if similarity(key_tokens, content_tokens(known)) > MAX_TOKEN_OVERLAP:
                return False
    return True


# ---------------------------------------------------------------------------
# Pool persistence
# ---------------------------------------------------------------------------

def known_elsewhere(philosopher: str, path: Path | None = None) -> set[str]:
    """Canonical quotes already attributed to a DIFFERENT thinker.

    The per-philosopher avoid set cannot catch cross-attribution: a 2026-08-01
    top-up handed Blaise Pascal "Those who can make you believe absurdities can
    make you commit atrocities.", which is Voltaire's and was already in
    Voltaire's pool. Publishing one thinker's line under another's name is the
    same misattribution failure as inventing one, so every generation path
    subtracts this set.
    """
    from fetcher import PHILOSOPHER_QUOTES

    key = philosopher.lower()
    out: set[str] = set()
    for name, rows in load_pool(path).items():
        if name != key:
            out |= canon_set(r["quote"] for r in rows)
    for name, quote_list in PHILOSOPHER_QUOTES.items():
        if name != key:
            out |= canon_set(quote_list)
    return out


def load_pool(path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Read the generated-quote pool. Malformed/missing file yields {}."""
    target = Path(path) if path is not None else QUOTE_POOL_PATH
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, OSError) as e:
        log.warning("quote pool unreadable (%s); starting empty.", e)
        return {}
    if not isinstance(data, dict):
        return {}
    # Shape-check every row rather than trusting the file.
    out: dict[str, list[dict[str, Any]]] = {}
    for name, rows in data.items():
        if not isinstance(rows, list):
            continue
        clean = [
            r for r in rows
            if isinstance(r, dict) and isinstance(r.get("quote"), str) and r["quote"].strip()
        ]
        if clean:
            out[str(name).lower()] = clean
    return out


def save_pool(pool: dict[str, list[dict[str, Any]]], path: Path | None = None) -> None:
    """Atomically persist the pool (tmp + replace, same pattern as state.py)."""
    target = Path(path) if path is not None else QUOTE_POOL_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(pool, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
    except OSError as e:
        # The pool is an enhancement; losing a write must not abort a reel.
        log.warning("could not persist quote pool: %s", e)


def add_to_pool(
    philosopher: str,
    entries: Sequence[dict[str, Any]],
    *,
    path: Path | None = None,
) -> int:
    """Merge new {quote, theme} rows into the persisted pool. Returns count added."""
    if not entries:
        return 0
    pool = load_pool(path)
    key = philosopher.lower()
    existing = pool.setdefault(key, [])
    seen = canon_set(r["quote"] for r in existing)
    added = 0
    for row in entries:
        c = canon(row.get("quote", ""))
        if not c or c in seen:
            continue
        existing.append({
            "quote": row["quote"].strip(),
            "theme": row.get("theme") or _DEFAULT_THEME,
            "source": row.get("source", "groq"),
            "added": date.today().isoformat(),
        })
        seen.add(c)
        added += 1
    if added:
        save_pool(pool, path)
    return added


# ---------------------------------------------------------------------------
# Theme classification + reward
# ---------------------------------------------------------------------------

def _keyword_theme(quote: str) -> str:
    """Offline theme guess. Never raises, always returns a valid THEME."""
    words = set(canon(quote).split())
    for theme, markers in _THEME_KEYWORDS.items():
        if any(m in words for m in markers):
            return theme
    return _DEFAULT_THEME


def classify_theme(quote: str, philosopher: str = "") -> str:
    """Assign a quote to one of THEMES.

    Tries Groq for a semantic read, falls back to keyword matching. Always
    returns a member of THEMES so a bad model response can never poison the
    bandit's arm space with an unknown arm.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _keyword_theme(quote)
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        resp = models.chat(
            client, models.FAST,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the philosophical quote into exactly ONE theme.\n"
                        "Allowed themes: " + ", ".join(THEMES) + "\n"
                        "Reply with the theme word only. No punctuation, no explanation."
                    ),
                },
                {"role": "user", "content": f"{philosopher}: {quote}\n\nTheme:"},
            ],
            max_tokens=8,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip().lower().strip(".")
        if raw in THEMES:
            return raw
        log.debug("classify_theme: unknown theme %r; using keyword fallback", raw)
    except Exception as e:  # noqa: BLE001 - external boundary
        log.debug("classify_theme: %s; using keyword fallback", e)
    return _keyword_theme(quote)


def theme_stats(entries: list[dict[str, Any]] | None = None) -> dict[str, dict[str, float]]:
    """Mean engagement reward per theme, straight from the upload ledger."""
    return bandit.arm_stats(THEMES, arm_field="theme", entries=entries)


def pick_theme(
    philosopher: str,
    post_count: int,
    *,
    entries: list[dict[str, Any]] | None = None,
    epsilon: float | None = None,
) -> str:
    """Choose which theme to publish next.

    Delegates to bandit.pick, so the Phase-1 contract holds: with no reward data
    in the ledger this is exactly THEMES[post_count % len(THEMES)], a plain
    round-robin that samples every theme evenly. Once insights accrue it biases
    toward the highest-mean-reward theme with seeded epsilon exploration.
    """
    return bandit.pick(
        THEMES,
        key=philosopher,
        post_count=post_count,
        arm_field="theme",
        entries=entries,
        epsilon=epsilon,
    )


def top_exemplars(
    philosopher: str,
    *,
    theme: str | None = None,
    limit: int = 4,
    entries: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Highest-reward quotes already published, to few-shot the generator.

    Prefers this philosopher's own winners, then widens to the whole account so
    a cold philosopher still gets steered by what works. Rows without a
    derivable reward are ignored (not treated as reward 0).
    """
    import ledger

    rows = entries if entries is not None else ledger.load_entries()
    scored: list[tuple[float, int, str]] = []
    for row in rows:
        quote = row.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            continue
        r = bandit.reward(row.get("insights"))
        if r is None:
            continue
        if theme and row.get("theme") != theme:
            continue
        # Same philosopher ranks ahead of the rest of the account.
        priority = 0 if row.get("philosopher", "").lower() == philosopher.lower() else 1
        scored.append((r, priority, quote.strip()))
    if not scored:
        return []
    scored.sort(key=lambda t: (t[1], -t[0]))
    out: list[str] = []
    seen: set[str] = set()
    for _, _, quote in scored:
        c = canon(quote)
        if c in seen:
            continue
        seen.add(c)
        out.append(quote)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Groq generation
# ---------------------------------------------------------------------------

# Static prefix kept verbatim so the Groq prompt cache stays warm across calls
# (all interpolated variables live in the user message). Same discipline as
# fetcher._SLOGAN_SYSTEM_PROMPT.
_GENERATE_SYSTEM_PROMPT = """You supply quotations for a philosophy Instagram account.

You are a RECALL tool, not an author. Return only quotations you actually know to be documented from the named thinker's published writing, letters, or reliably recorded speech.

Rules:
- Output STRICT JSON: {"quotes": ["...", "..."]}. No prose, no markdown fence.
- Each quote must be a real, documented line from that thinker. If you are not confident a line is genuinely theirs, omit it.
- Return FEWER quotes rather than inventing any. An empty list is an acceptable, correct answer.
- Quote the line IN FULL and end it with terminal punctuation. Never truncate. A clause that depends on a missing second half is worthless:
    WRONG: "Those who can make you believe absurdities"
    RIGHT: "Those who can make you believe absurdities can make you commit atrocities."
    WRONG: "You must have chaos within you"
    RIGHT: "You must have chaos within you to give birth to a dancing star."
- 26 to 220 characters each. Self-contained, no surrounding context needed.
- No attribution inside the quote text. No author name, no source title, no dates.
- Plain ASCII punctuation. No em dashes.
- Do not repeat any quote listed as already used."""


_VERIFY_SYSTEM_PROMPT = """You are a strict attribution checker for philosophical quotations.

For each numbered candidate, answer YES if BOTH hold:
  (a) the line is genuinely documented in the named thinker's published writing, letters, or reliably recorded speech, AND
  (b) it is the COMPLETE quotation, not a truncated fragment missing its second half.

Name the work it comes from in "source". If you cannot name a work, answer NO.

Answer NO for anything you merely find plausible, stylistically similar, generic, or cannot place in their work. A wrong YES publishes a fabricated quotation under a real person's name, so NO is always the safe answer when unsure.

Be especially suspicious of a line that ALMOST matches something you know. A reworded, condensed, or "improved" version of a real passage is NOT the quotation -- it is a fabrication wearing its clothes. If the wording you recall differs at all, answer NO.

NO examples:
  Camus, "Defiance is the only way"                      -> NO, invented
  Camus, "The absurd man says yes and his effort is unending forever."
                                                         -> NO, reworded; the real line reads "...his effort will henceforth be unceasing"
  Aurelius, "Discipline is the bridge to achievement."   -> NO, modern self-help, not Meditations
  Voltaire, "Those who can make you believe absurdities" -> NO, truncated, drops "can make you commit atrocities"
  Nietzsche, "You must have chaos within you"            -> NO, truncated, drops "to give birth to a dancing star"

Output STRICT JSON: {"verdicts": [{"n": 1, "ok": true, "source": "The Rebel"}, {"n": 2, "ok": false, "source": ""}]}
No prose, no markdown fence."""


class VerificationUnavailable(RuntimeError):
    """The attribution check could not run (no key, network, malformed response).

    Distinct from "every candidate was rejected". Publishing paths treat both
    the same and fall back safely, but any path that DELETES on a negative
    verdict must tell them apart: a transient 429 that reads as "all rejected"
    would otherwise wipe the pool.
    """


def _verify_once(
    philosopher: str,
    candidates: Sequence[str],
    *,
    temperature: float,
    on_unavailable,
) -> list[bool]:
    """One verification sample. See verify_quotes() for the contract.

    Distinguishes "the model ruled NO" from "the model never ruled". A reply
    that omits candidates (truncated JSON, dropped entries) is an incomplete
    response, NOT a rejection: under unanimity voting, silently reading an
    omission as NO would let one truncated sample permanently reject a genuine
    quote, which is the same failure mode as reading a 429 as "all fabricated".
    Incomplete replies route through on_unavailable instead.
    """
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(candidates, 1))
    try:
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = models.chat(
            client, models.SMART,
            messages=[
                {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Thinker: {philosopher}\n\n{numbered}"},
            ],
            # Each sample must emit a full verdict array; too small a budget
            # truncates the JSON and looks like an omission.
            max_tokens=180 * len(candidates) + 200,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        payload = json.loads((resp.choices[0].message.content or "").strip())
        verdicts = payload.get("verdicts") if isinstance(payload, dict) else None
        if not isinstance(verdicts, list):
            return on_unavailable("malformed response")
    except VerificationUnavailable:
        raise
    except Exception as e:  # noqa: BLE001 - external boundary
        return on_unavailable(str(e))

    approved = [False] * len(candidates)
    ruled: set[int] = set()
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        idx = v.get("n")
        if not (isinstance(idx, int) and 1 <= idx <= len(candidates)):
            continue
        ruled.add(idx - 1)
        if v.get("ok") is not True:
            continue
        # Source-grounding: an approval with no nameable work is the signature
        # of a confabulated YES.
        source = v.get("source")
        if not (isinstance(source, str) and source.strip()):
            log.info("verify_quotes: YES without a source; rejecting %.60r",
                     candidates[idx - 1])
            continue
        approved[idx - 1] = True

    if len(ruled) < len(candidates):
        return on_unavailable(
            f"incomplete verdicts ({len(ruled)}/{len(candidates)} ruled on)"
        )
    return approved


def verify_quotes(
    philosopher: str,
    candidates: Sequence[str],
    *,
    fail_closed: bool = True,
    votes: int = 1,
) -> list[bool]:
    """Second-pass attribution check on generated candidates.

    fail_closed=True  (default, for publishing): infrastructure failure returns
                      all-False, so nothing unverified can ever ship.
    fail_closed=False (for destructive callers): infrastructure failure raises
                      VerificationUnavailable instead, so a transient error is
                      never mistaken for a genuine rejection.

    Approves a line only if EVERY sample says yes. `votes` defaults to 1
    because self-consistency was measured and did NOT help: the hardest failure
    is a near-paraphrase of a real passage ("The absurd man says yes and his
    effort is unending forever." under Camus, where the real line ends "...his
    effort will henceforth be unceasing"), and the model believes that one
    STABLY across samples. Voting is 3x the cost for zero measured gain. The
    knob is kept for experimentation; raising it will not buy accuracy.

    Measured on 18 known-genuine and 8 known-fabricated quotes:
        strict prompt, temp 0   -> 50-56% recall, 0% false-accept  (shipped)
        + YES few-shot examples -> 61% recall, 12% false-accept    (rejected)

    Recall was deliberately left low. Yield is raised by asking the generator
    for MORE candidates, never by loosening this gate: a false accept publishes
    a fabricated quotation under a real person's name, while a false reject
    costs nothing but an extra API call.
    """
    def _unavailable(reason: str) -> list[bool]:
        if fail_closed:
            log.warning("verify_quotes: %s; rejecting all.", reason)
            return [False] * len(candidates)
        raise VerificationUnavailable(reason)

    if not candidates:
        return []
    if not os.environ.get("GROQ_API_KEY"):
        return _unavailable("GROQ_API_KEY missing")

    n_votes = max(1, votes)
    # Single vote wants determinism; only multi-vote needs sampling jitter to
    # carry information across samples.
    temperature = 0.0 if n_votes == 1 else 0.3
    approved = [True] * len(candidates)
    for _ in range(n_votes):
        sample = _verify_once(
            philosopher, candidates,
            temperature=temperature,
            on_unavailable=_unavailable,  # closure already encodes fail_closed
        )
        approved = [a and b for a, b in zip(approved, sample)]
        if not any(approved):
            break  # nothing left to confirm; skip the remaining samples
    for quote, ok in zip(candidates, approved):
        if not ok:
            log.debug("verify_quotes: not unanimous; rejecting %.60r", quote)
    return approved


def request_candidates(
    philosopher: str,
    *,
    theme: str,
    exemplars: Sequence[str] = (),
    n: int = 6,
) -> list[str]:
    """Raw Groq recall call. Returns candidate quote strings, [] on any failure.

    Split out from generate_quotes() purely as a seam: it isolates the single
    network boundary so the filtering/verification logic around it is testable
    without mocking the Groq SDK. Never raises.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log.warning("request_candidates: GROQ_API_KEY missing; cannot extend pool.")
        return []

    parts = [f"Thinker: {philosopher}", f"Desired theme: {theme}"]
    if exemplars:
        parts.append(
            "Quotes from this account that earned the most engagement (match this "
            "register, do NOT reuse them):\n"
            + "\n".join(f"- {q}" for q in exemplars)
        )
    parts.append(f"Return up to {n} quotes as JSON.")

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        resp = models.chat(
            client, models.SMART,
            messages=[
                {"role": "system", "content": _GENERATE_SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(parts)},
            ],
            max_tokens=700,
            temperature=0.4,  # low: we want recall, not invention
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 - external boundary
        log.warning("request_candidates: Groq call failed: %s", e)
        return []

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        log.warning("request_candidates: unparseable response (%s): %.120r", e, raw)
        return []

    items = payload.get("quotes") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        log.warning("request_candidates: response missing 'quotes' list.")
        return []
    return [i.strip() for i in items if isinstance(i, str) and i.strip()]


def generate_quotes(
    philosopher: str,
    *,
    theme: str,
    exemplars: Sequence[str] = (),
    avoid: set[str] | None = None,
    # Verifier recall is intentionally low (39% on the current model, see
    # models.SMART). Yield is raised by asking for more candidates, never by
    # loosening the attribution gate.
    n: int = 16,
    verify: bool = True,
) -> list[dict[str, Any]]:
    """Ask Groq for fresh documented quotes on `theme`, steered by `exemplars`.

    `exemplars` are the account's highest-reward past quotes: they show the model
    the register that actually performs rather than describing it abstractly.
    Every returned line must clear is_acceptable() before it is kept, and the
    theme is attached without a second classification round-trip.

    Returns [] on any failure -- caller must have a fallback.
    """
    candidates = request_candidates(philosopher, theme=theme, exemplars=exemplars, n=n)
    if not candidates:
        return []

    avoid = avoid or set()
    kept: list[dict[str, Any]] = []
    local_avoid = set(avoid)
    for item in candidates:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not is_acceptable(text, avoid=local_avoid):
            log.debug("generate_quotes: rejected %.80r", text)
            continue
        local_avoid.add(canon(text))
        kept.append({"quote": text, "theme": theme, "source": "groq"})

    if kept and verify:
        approved = verify_quotes(philosopher, [r["quote"] for r in kept])
        rejected = [r["quote"] for r, ok in zip(kept, approved) if not ok]
        for quote in rejected:
            log.info("generate_quotes: attribution check REJECTED %.80r", quote)
        kept = [r for r, ok in zip(kept, approved) if ok]

    log.info(
        "generate_quotes(%s, theme=%s): kept %d/%d candidates.",
        philosopher, theme, len(kept), len(candidates),
    )
    return kept


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _lru_replay(pool_rows: list[dict[str, Any]], used_quotes: Sequence[str]) -> dict[str, Any]:
    """Pick the pool quote whose last use is furthest back.

    The floor of the supply chain. Guarantees that when nothing fresh exists and
    generation is unavailable, repeats are spaced as widely as the pool allows
    instead of collapsing onto a single quote the way all_quotes[0] did.
    """
    last_seen: dict[str, int] = {}
    for i, q in enumerate(used_quotes):
        c = canon(q)
        if c:
            last_seen[c] = i
    # Never-used rows sort first (-1); otherwise oldest use wins. Ties break on
    # pool order for determinism.
    return min(
        pool_rows,
        key=lambda r: (last_seen.get(canon(r["quote"]), -1), pool_rows.index(r)),
    )


def select_quote(
    philosopher: str,
    used_quotes: Sequence[str],
    *,
    post_count: int = 0,
    entries: list[dict[str, Any]] | None = None,
    allow_generation: bool = True,
    pool_path: Path | None = None,
) -> dict[str, Any]:
    """Return {"quote", "theme", "reframed", "source"} for the next reel.

    Walks the supply chain documented at the top of this module. `reframed` stays
    True only for a replayed quote, preserving its original meaning (this line is
    not brand new) for state.json and the caption layer.
    """
    from fetcher import PHILOSOPHER_QUOTES

    key = philosopher.lower()
    static_rows = [
        {"quote": q, "theme": _keyword_theme(q), "source": "static"}
        for q in PHILOSOPHER_QUOTES.get(key, [])
    ]
    pool_rows = static_rows + load_pool(pool_path).get(key, [])

    if not pool_rows:
        log.error("select_quote: no quotes at all for %s.", philosopher)
        return {
            "quote": "The unexamined life is not worth living.",
            "theme": _DEFAULT_THEME,
            "reframed": True,
            "source": "fallback",
        }

    used = canon_set(used_quotes)
    fresh = [r for r in pool_rows if canon(r["quote"]) not in used]
    theme = pick_theme(philosopher, post_count, entries=entries)

    # 1. fresh quote on the theme the bandit wants
    on_theme = [r for r in fresh if r.get("theme") == theme]
    if on_theme:
        chosen = on_theme[0]
        return {**chosen, "reframed": False}

    # 2. any fresh quote (theme unavailable in stock)
    if fresh:
        chosen = fresh[0]
        log.info(
            "select_quote: no fresh %r quote for %s; serving %r instead.",
            theme, philosopher, chosen.get("theme"),
        )
        return {**chosen, "reframed": False}

    # 3. pool exhausted -- this is the condition that silently froze the
    #    pipeline for 50+ runs. It is an ERROR, and it must be loud.
    log.error(
        "QUOTE POOL EXHAUSTED for %s: all %d known quotes already published "
        "(post_count=%d). Attempting Groq top-up.",
        philosopher, len(pool_rows), post_count,
    )

    if allow_generation:
        # Belt and braces around the whole generation attempt: step 4 (LRU
        # replay) is the floor of this module and must survive ANY failure
        # here, including an unexpected one from the ledger read or the Groq
        # SDK. A crash at this point would take down the reel entirely.
        try:
            exemplars = top_exemplars(philosopher, theme=theme, entries=entries)
            if not exemplars:
                exemplars = top_exemplars(philosopher, entries=entries)
            generated = generate_quotes(
                philosopher,
                theme=theme,
                exemplars=exemplars,
                avoid=(used
                       | canon_set(r["quote"] for r in pool_rows)
                       | known_elsewhere(philosopher, pool_path)),
            )
        except Exception as e:  # noqa: BLE001 - never lose the LRU fallback
            log.error("select_quote: generation raised for %s: %s", philosopher, e)
            generated = []
        if generated:
            add_to_pool(philosopher, generated, path=pool_path)
            chosen = generated[0]
            log.info(
                "select_quote: extended %s pool by %d; serving a generated quote.",
                philosopher, len(generated),
            )
            return {**chosen, "reframed": False}
        log.error(
            "select_quote: Groq top-up produced nothing usable for %s; "
            "falling back to LRU replay.", philosopher,
        )

    # 4. floor: replay the least-recently-used quote
    chosen = _lru_replay(pool_rows, used_quotes)
    log.warning(
        "select_quote: REPLAYING least-recently-used quote for %s.", philosopher,
    )
    return {**chosen, "reframed": True}
