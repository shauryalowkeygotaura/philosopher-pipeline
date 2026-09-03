"""
bandit.py -- epsilon-greedy arm picker for caption hooks / slogans.

The pipeline has a fixed set of opening "hook" lines (pipeline.HOOKS). Which
hook a reel opens with is an arm of a multi-armed bandit; the reward is the
engagement the resulting post earns (read from the local ledger, populated by a
delayed insights pull). This module decides which arm to play.

Phase 1  no reward data yet -> return the legacy deterministic round-robin pick
         arms[post_count % len(arms)]. Reel output stays BYTE-IDENTICAL.
Phase 2  delayed insights present in the ledger -> bias toward the
         highest-mean-reward arm, with deterministic epsilon exploration that
         falls back to the round-robin pick so every arm keeps being sampled.
Phase 3  the LIVE insights pull (instagrapi insights_media) is account-type
         gated and OFF by default. live_insights_enabled() is the env gate;
         the pull lives in insights.py and is never auto-invoked here.

Determinism: exploration is seeded from (key, post_count) via SHA-256, not
Python's salted hash(), so a given philosopher+post_count always resolves the
same way across processes. This keeps runs reproducible and testable.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Callable, Sequence

import ledger

# Default exploration rate once reward data exists. Override via env for tuning.
DEFAULT_EPSILON = float(os.getenv("PHILOSOPHER_BANDIT_EPSILON", "0.2"))


def live_insights_enabled() -> bool:
    """Phase 3 gate.

    Live insight pulls require a Business/Creator account (the Instagram
    insights endpoint is account-type gated) AND an explicit opt-in. Default
    OFF: when this returns False nothing in the loop makes a network call.
    """
    return os.getenv("PHILOSOPHER_LIVE_INSIGHTS", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def reward(insights: Any) -> float | None:
    """Map an insights dict to a single scalar reward, or None if no signal.

    Weighted to match the pipeline's CTA priorities (saves > shares > the rest):
    saves*3 + shares*2 + likes + comments. Falls back to reach/plays when only
    a coarse metric is available. Unknown / non-numeric inputs return None so
    the arm is treated as "no data yet" rather than reward 0.
    """
    if not isinstance(insights, dict):
        return None

    def _num(*keys: str) -> float:
        for k in keys:
            v = insights.get(k)
            if isinstance(v, bool):  # guard: bool is a subclass of int
                continue
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0

    saves = _num("saved", "saves", "save_count")
    shares = _num("shared", "shares", "share_count", "reshare_count")
    likes = _num("like_count", "likes")
    comments = _num("comment_count", "comments")
    weighted = saves * 3.0 + shares * 2.0 + likes + comments
    if weighted > 0:
        return weighted

    coarse = _num("reach", "play_count", "plays", "video_view_count", "impression_count", "impressions")
    return coarse if coarse > 0 else None


def _seed_rng(key: str, post_count: int) -> float:
    """Deterministic uniform in [0,1) from (key, post_count) via SHA-256."""
    digest = hashlib.sha256(f"{key}|{post_count}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def arm_stats(
    arms: Sequence[str],
    *,
    arm_field: str = "hook",
    entries: list[dict[str, Any]] | None = None,
    reward_fn: Callable[[Any], float | None] | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate reward observations per arm value from the ledger.

    Only rows whose arm value is one of `arms` and that carry insights with a
    derivable reward are counted. Returns {arm: {"n": count, "mean": avg}}.
    """
    rows = entries if entries is not None else ledger.load_entries()
    rfn = reward_fn or reward
    arm_set = set(arms)
    acc: dict[str, list[float]] = {}
    for row in rows:
        arm_value = row.get(arm_field)
        if arm_value not in arm_set:
            continue
        r = rfn(row.get("insights"))
        if r is None:
            continue
        acc.setdefault(arm_value, []).append(r)
    return {
        arm: {"n": float(len(vals)), "mean": sum(vals) / len(vals)}
        for arm, vals in acc.items()
        if vals
    }


def pick(
    arms: Sequence[str],
    *,
    key: str,
    post_count: int,
    arm_field: str = "hook",
    entries: list[dict[str, Any]] | None = None,
    epsilon: float | None = None,
    reward_fn: Callable[[Any], float | None] | None = None,
) -> str:
    """Choose an arm.

    Contract: when no arm has any reward observation, returns exactly
    arms[post_count % len(arms)] (the legacy deterministic pick) so output is
    unchanged until data accrues. Once reward data exists, plays epsilon-greedy
    with deterministic, seeded exploration.
    """
    if not arms:
        raise ValueError("pick() requires at least one arm")

    baseline = arms[post_count % len(arms)]

    stats = arm_stats(arms, arm_field=arm_field, entries=entries, reward_fn=reward_fn)
    if not stats:
        # Phase 1: no learning signal -> byte-identical legacy behaviour.
        return baseline

    eps = DEFAULT_EPSILON if epsilon is None else epsilon
    # Explore with probability eps: fall back to the round-robin baseline so
    # every arm (including ones never yet sampled) keeps getting played.
    if _seed_rng(key, post_count) < eps:
        return baseline

    # Exploit: highest mean reward among observed arms; ties broken by the arm's
    # position in `arms` for deterministic, reproducible selection.
    best_arm = baseline
    best_mean = float("-inf")
    for arm in arms:  # iterate in declared order for stable tie-breaking
        s = stats.get(arm)
        if s is None:
            continue
        if s["mean"] > best_mean:
            best_mean = s["mean"]
            best_arm = arm
    return best_arm


def pick_hook(
    philosopher: str,
    post_count: int,
    hooks: Sequence[str],
    *,
    entries: list[dict[str, Any]] | None = None,
    epsilon: float | None = None,
) -> str:
    """Convenience wrapper: pick a caption hook for `philosopher`.

    Drop-in for the old `hooks[post_count % len(hooks)]` line. Identical result
    until ledger insights exist for this hook set.
    """
    return pick(
        hooks,
        key=philosopher,
        post_count=post_count,
        arm_field="hook",
        entries=entries,
        epsilon=epsilon,
    )


def loop_status(arms: Sequence[str] | None = None, entries=None) -> dict:
    """Is the bandit actually learning, or still round-robinning?

    This exists because it was not obvious that it was not. The bandit is well
    built - phased, deterministic, safe by default - and it has spent its entire
    life in Phase 1, because no ledger row ever received a reward:
    live_insights_enabled() is off by default and insights.refresh_pending() is
    never auto-invoked. A mechanism that silently does nothing looks exactly
    like a mechanism that works.

    Same check as format-engine's feedback.status(). Cheap, local, no network.
    """
    rows = entries if entries is not None else ledger.load_entries()
    # Arms are passed in rather than imported: HOOKS lives in pipeline.py, and
    # importing it here would make bandit depend on the module that depends on
    # bandit. With no arms given, count reward-bearing rows directly - the
    # question "is anything learning" does not need the arm list to answer.
    # Rewards present in the ledger REGARDLESS of which arm they belong to.
    # The gap between this and `observed` is the whole diagnostic: 134 rewards
    # that match no current arm look identical to no rewards at all.
    total_rewards = sum(1 for r in rows if isinstance(r, dict)
                        and reward(r.get("insights")) is not None)
    if arms:
        stats = arm_stats(arms, entries=rows)
        observed = sum(s.get("n", 0) for s in stats.values())
        arms_with_data = sum(1 for s in stats.values() if s.get("n", 0) > 0)
    else:
        observed = total_rewards
        arms_with_data = len({r.get("hook") for r in rows
                              if isinstance(r, dict)
                              and reward(r.get("insights")) is not None})
    # Which hook texts actually appear in the ledger, so a mismatch is visible
    # rather than showing up as a silent zero.
    seen = sorted({(r.get("hook") or "")[:38] for r in rows
                   if isinstance(r, dict) and r.get("hook")})[:5]
    return {
        "ledger_rows": len(rows),
        "ledger_hook_samples": seen,
        "reward_observations": observed,
        "arms_with_data": arms_with_data,
        "live_insights_enabled": live_insights_enabled(),
        "phase": 2 if observed else 1,
        "rewards_in_ledger": total_rewards,
        "verdict": ("learning" if observed else
                    f"ORPHANED - the ledger holds {total_rewards} reward rows and "
                    "NONE match a hook in the current list, so none of them count. "
                    "The hook wording changed after those posts, and a bandit keyed "
                    "on exact text cannot connect the two. Learning restarts from "
                    "zero unless the old wording is restored."
                    if total_rewards else
                    "ORPHANED - the ledger HAS rewards, but none belong to an arm "
                    "in the current list, so none of them count. The hook text "
                    "changed after those posts were made. Either restore the old "
                    "wording or accept that learning restarts from zero."
                    if observed and not arms else
                    "OPEN - Phase 1 round-robin. No reward rows, so the bandit "
                    "picks in rotation and no arm is ever preferred. Restore "
                    "Instagram auth, set PHILOSOPHER_LIVE_INSIGHTS=1, then run "
                    "insights.refresh_pending() a few days after posting."),
    }
