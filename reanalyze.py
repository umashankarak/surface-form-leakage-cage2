#!/usr/bin/env python3
"""
Gate 2 REANALYSIS -- read-only on runs/gate2.jsonl.

The reward metric saturates: B_line's fast fixed path can't be stopped in 30
steps by an imperfect defender, so competent and incompetent models both floor,
and accumulated reward mostly reflects attacker-seed luck. This script recomputes
the L0-vs-L1 contrast under metrics that DON'T saturate, using data already
collected -- no new episodes.

Metrics, per rung, L0 vs L1 (paired by seed where possible):

  reward        the existing metric, for comparison
  reach_rate    P(attacker compromised the crown jewel this episode). Binary,
                directly on-question: did disguising names let red through more?
  survival      mean turn of first large-penalty spike (proxy for time-to-
                compromise). Longer = defender held out longer. More sensitive
                than a binary because it separates "lost at turn 5" from
                "lost at turn 25".
  ep_len        mean episode length. Diagnostic: short episodes = repeat-abort
                artifact (a model that loops gets terminated early and dodges
                penalty -- inflating its reward score dishonestly).

For each metric the drop is L1 - L0 with a proper CI. For reach (binary) the
per-episode series is 0/1 and the same t-based CI is an adequate approximation
at N=40; a proportion test would refine it but won't change the conclusion.
"""
import json
import math
import statistics
import sys

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

PATH = sys.argv[1] if len(sys.argv) > 1 else "runs/gate2.jsonl"
# Impact on the crown jewel is a large single-turn penalty; use a threshold to
# detect the compromise turn from per-turn rewards.
COMPROMISE_PENALTY = -8.0


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def survival_turn(rec, horizon=30):
    """First turn whose reward drops below COMPROMISE_PENALTY; else horizon."""
    for t in rec.get("turns", []):
        if t.get("reward", 0) <= COMPROMISE_PENALTY:
            return t["turn"]
    return horizon


def episode_metrics(rec):
    return {
        "reward": rec.get("total_reward", 0.0),
        "reach": 1.0 if rec.get("red_reached_goal") else 0.0,
        "survival": survival_turn(rec),
        "ep_len": rec.get("n_turns", 0),
        "aborted": 1.0 if rec.get("abort_reason") else 0.0,
    }


def ci_drop(l0, l1):
    """Return (mean0, mean1, drop, ci_halfwidth, p). drop = mean1 - mean0."""
    n0, n1 = len(l0), len(l1)
    if n0 < 2 or n1 < 2:
        return (float("nan"),) * 5
    m0, m1 = statistics.mean(l0), statistics.mean(l1)
    v0, v1 = statistics.variance(l0), statistics.variance(l1)
    drop = m1 - m0
    se = math.sqrt(v0 / n0 + v1 / n1)
    if se == 0:
        return m0, m1, drop, 0.0, (0.0 if drop == 0 else 0.0)
    if HAVE_SCIPY:
        df = (v0 / n0 + v1 / n1) ** 2 / (
            (v0 / n0) ** 2 / (n0 - 1) + (v1 / n1) ** 2 / (n1 - 1))
        ci = stats.t.ppf(0.975, df) * se
        p = 2 * (1 - stats.t.cdf(abs(drop) / se, df))
    else:
        ci, p = 1.96 * se, float("nan")
    return m0, m1, drop, ci, p


def main():
    rows = load(PATH)
    if not rows:
        print(f"no data in {PATH}")
        return 1

    # organise: cells[(model, level)] = list of per-episode metric dicts
    cells = {}
    for r in rows:
        cells.setdefault((r["model"], r["level"]), []).append(episode_metrics(r))
    models = sorted({m for m, _ in cells})

    floor = -217.5  # passive-blue floor from Gate 2

    print("=" * 92)
    print("GATE 2 REANALYSIS -- multiple metrics on existing data (read-only)")
    print("=" * 92)
    print(f"source: {PATH}")
    print(f"crown-jewel compromise detected at per-turn reward <= {COMPROMISE_PENALTY}")
    print()

    for metric, better, note in [
        ("reward",   "higher", "existing metric; saturates at floor ~-217"),
        ("reach",    "lower",  "P(attacker compromised crown jewel); binary, on-question"),
        ("survival", "higher", "mean turn of compromise; higher = held out longer"),
        ("ep_len",   "-",      "diagnostic: short => repeat-abort artifact"),
    ]:
        arrow = {"higher": "(higher=better defence)",
                 "lower": "(lower=better defence)",
                 "-": "(diagnostic only)"}[better]
        print(f"--- {metric.upper()}  {arrow}  -- {note}")
        print(f"    {'model':16s} {'L0':>8s} {'L1':>8s} {'drop':>8s} {'95%CI':>9s} {'p':>7s}  {'note':>10s}")
        for m in models:
            c0 = cells.get((m, "L0"), [])
            c1 = cells.get((m, "L1"), [])
            if len(c0) < 2 or len(c1) < 2:
                continue
            s0 = [e[metric] for e in c0]
            s1 = [e[metric] for e in c1]
            m0, m1, drop, ci, p = ci_drop(s0, s1)
            tag = ""
            if metric == "reward" and (m0 <= floor + 5):
                tag = "FLOOR"
            if metric == "ep_len" and m0 < 15:
                tag = "ABORTS"
            sig = "" if (ci != ci or p != p) else ("sig" if abs(drop) > ci else "ns")
            print(f"    {m:16s} {m0:8.2f} {m1:8.2f} {drop:+8.2f} "
                  f"{'+/-'+format(ci,'.2f'):>9s} {p:7.3f}  {sig:>4s} {tag}")
        print()

    print("=" * 92)
    print("HOW TO READ THIS")
    print("  - If REACH or SURVIVAL shows a clean pattern (drops shrinking as")
    print("    model size grows) where REWARD did not, that metric was the")
    print("    problem, and it becomes the study's measure.")
    print("  - Watch EP_LEN: any rung with ep_len < 15 is repeat-aborting, so its")
    print("    REWARD is inflated (early exit dodges penalty). Those reward numbers")
    print("    are not trustworthy regardless of significance.")
    print("  - A rung at FLOOR on reward can still be informative on reach/survival")
    print("    if it sometimes delays the attacker -- that's the point of switching.")
    print("  - If reach AND survival are BOTH flat across rungs: honest null. The")
    print("    effect isn't visible at these sizes, arrived at properly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())