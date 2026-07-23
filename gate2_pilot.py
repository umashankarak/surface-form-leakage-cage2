#!/usr/bin/env python3
"""
GATE 2 -- does the retrieval drop exist at a rung anyone deploys?

Gate 1 proved the twins are difficulty-matched with a live signal. Gate 2 asks
the question the whole study hinges on: when a real LLM plays L0 vs L1, does its
performance drop -- and does the drop appear at a model size someone would
actually run, not only at 0.5B?

WHAT THIS MEASURES, AND THE TRAPS IT GUARDS AGAINST
---------------------------------------------------
For each rung (model) it runs L0 (canonical) and L1 (renamed, semantics
preserved), N seeds each, and reports:

  drop         = mean(L1) - mean(L0). The headline. Negative = worse on twin.

  FLOOR CHECK  A model that scores ~floor on BOTH conditions is uninformative
               (it can't defend at all, so there's no performance to lose).
               This is the real failure mode -- NOT "model too weak". The
               measurement is a within-model contrast, so a weak model is fine
               as long as it clears the floor on L0. Flagged per rung.

  PARSE FAILS  If a small model fails to FORMAT actions while a big one doesn't,
               part of any measured drop is formatting competence, not
               retrieval. Reported per rung per condition so the two can be
               separated. A rung with high parse-failure rate is suspect.

  SIGNAL LIVE  red_reached_goal rate per condition. Sanity that the crown-jewel
               threat is actually present in these episodes (should track
               Gate 1's ~16%). If it's ~0, the renamed fact was irrelevant and
               a null drop means nothing.

INTERPRETATION
  drop clearly < 0 at a deployable rung, low parse-fails, signal live
        -> the effect is real. Proceed to the full grid (Phase 3).
  drop ~ 0 everywhere
        -> no contamination signal at these sizes; the benchmark isn't inflated
           for these models. Honest null. Stop or rethink.
  drop < 0 ONLY at 0.5B-1B
        -> "tiny models are bad at security". Not a paper. Stop.

Usage:
  python3 gate2_pilot.py --rungs ollama:qwen2.5:3b ollama:qwen2.5:14b --n 10
  python3 gate2_pilot.py --rungs mock --n 5          # wiring test, no models
"""
import argparse
import os
import statistics
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

from cyborg_llm.env import default_scenario_path
from cyborg_llm.twins import load_scenario, make_twin, write_scenario, crown_jewel
from cyborg_llm.probes import make_bline_probe_class, patch_simulator_enterprise_check
from cyborg_llm.providers import get_provider
from cyborg_llm.runner import run_episode, append_record, completed_keys

STEPS = 30
DEFAULT_LEVELS = ["L0", "L1"]      # L1 defaults to the alt_a semantic scheme


def _floor_estimate(scenario, path, n=STEPS):
    """
    The passive-blue floor for THIS scenario: what an agent that only ever
    Monitors scores. A model near this on L0 has learned nothing.
    """
    import random
    import numpy as np
    from CybORG import CybORG
    from CybORG.Agents import BlueMonitorAgent

    patch_simulator_enterprise_check(scenario)
    vals = []
    for seed in range(min(n, 10)):
        random.seed(seed)
        np.random.seed(seed)
        cy = CybORG(path, "sim", agents={"Red": make_bline_probe_class(scenario)})
        obs = cy.reset(agent="Blue").observation
        blue = BlueMonitorAgent()
        tot = 0.0
        for _ in range(STEPS):
            a = blue.get_action(obs, cy.get_action_space("Blue"))
            r = cy.step(agent="Blue", action=a)
            obs = r.observation
            tot += r.reward
        vals.append(tot)
    return statistics.mean(vals)


def run_cell(provider_spec, scenario, path, level, n, out_path):
    """Run N episodes of one (model, level) cell. Resumable. Returns records."""
    provider = get_provider(provider_spec)
    model = getattr(provider, "model", provider.name)
    goal = crown_jewel(scenario)
    patch_simulator_enterprise_check(scenario)      # global; re-apply per scenario
    red_cls = make_bline_probe_class(scenario)

    done = completed_keys(out_path)
    records = []
    for seed in range(n):
        key = (model, level, seed)
        if key in done:
            continue
        rec = run_episode(provider, seed, level=level, scenario_path=path,
                          red_agent=red_cls, max_turns=STEPS,
                          scenario=scenario, crown_jewel_name=goal)
        append_record(out_path, rec)
        records.append(rec)
    # gather everything for this cell (including previously-done) from disk
    return _load_cell(out_path, model, level, n)


def _load_cell(out_path, model, level, n):
    import json
    rows = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("model") == model and r.get("level") == level and r.get("seed", -1) < n:
                    rows.append(r)
    return rows


def summarize(rows):
    rew = [r["total_reward"] for r in rows]
    pf = sum(r.get("parse_failures", 0) for r in rows)
    aborts = sum(1 for r in rows if r.get("abort_reason"))
    reach = sum(1 for r in rows if r.get("red_reached_goal")) / len(rows) if rows else 0.0
    return {
        "n": len(rows),
        "mean": statistics.mean(rew) if rew else float("nan"),
        "sd": statistics.stdev(rew) if len(rew) > 1 else 0.0,
        "rewards": rew,
        "parse_fails": pf,
        "aborts": aborts,
        "reach": reach,
    }


def drop_stats(l0_rewards, l1_rewards):
    """
    Is the drop distinguishable from zero? Returns (drop, ci_halfwidth, p).

    The whole point of Gate 2 at pilot N: a drop whose CI straddles zero is not
    yet evidence of anything. sd is ~42 for B_line episodes, so at N=10 the CI
    half-width is ~26 -- the size of the effect itself. This is what stops the
    read-out from calling noise a result.
    """
    n0, n1 = len(l0_rewards), len(l1_rewards)
    if n0 < 2 or n1 < 2:
        return float("nan"), float("nan"), float("nan")
    m0, m1 = statistics.mean(l0_rewards), statistics.mean(l1_rewards)
    v0, v1 = statistics.variance(l0_rewards), statistics.variance(l1_rewards)
    drop = m1 - m0
    se = (v0 / n0 + v1 / n1) ** 0.5
    try:
        from scipy import stats
        df = (v0 / n0 + v1 / n1) ** 2 / (
            (v0 / n0) ** 2 / (n0 - 1) + (v1 / n1) ** 2 / (n1 - 1)) if se > 0 else 1
        p = 2 * (1 - stats.t.cdf(abs(drop) / se, df)) if se > 0 else 1.0
        tcrit = stats.t.ppf(0.975, df)
    except ImportError:
        p, tcrit = float("nan"), 1.96
    return drop, tcrit * se, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", nargs="+", required=True,
                    help="provider specs, e.g. ollama:qwen2.5:3b ollama:qwen2.5:14b")
    ap.add_argument("--n", type=int, default=10, help="episodes per cell")
    ap.add_argument("--scheme", default="alt_a", help="L1 rename scheme (alt_a/alt_b/neutral)")
    ap.add_argument("--out", default="runs/gate2.jsonl")
    args = ap.parse_args()

    orig = load_scenario(default_scenario_path())
    goal = crown_jewel(orig)
    tmp = tempfile.mkdtemp(prefix="gate2_")
    paths = {"L0": os.path.join(tmp, "L0.yaml")}
    write_scenario(orig, paths["L0"])
    twin, _ = make_twin(orig, level="L1", scheme=args.scheme)
    paths["L1"] = os.path.join(tmp, "L1.yaml")
    write_scenario(twin, paths["L1"])
    scen = {"L0": orig, "L1": twin}

    floor = _floor_estimate(orig, paths["L0"])

    print("=" * 78)
    print("GATE 2 -- DOES THE RETRIEVAL DROP EXIST AT A DEPLOYABLE RUNG?")
    print("=" * 78)
    print(f"crown jewel = {goal} | L1 scheme = {args.scheme} | N = {args.n} | {STEPS} steps")
    print(f"passive-blue floor (score ~= this means the model learned nothing): {floor:.1f}")
    print(f"logging to {args.out} (resumable)")
    print()
    print(f"{'RUNG':22s} {'L0 mean':>9s} {'L1 mean':>9s} {'DROP':>8s} "
          f"{'95% CI':>13s} {'p':>6s} {'pf':>5s} {'reach':>6s}  FLAGS")
    print("-" * 88)

    results = []
    for spec in args.rungs:
        s = {}
        for lvl in DEFAULT_LEVELS:
            rows = run_cell(spec, scen[lvl], paths[lvl], lvl, args.n, args.out)
            s[lvl] = summarize(rows)
        drop, ci, p = drop_stats(s["L0"]["rewards"], s["L1"]["rewards"])

        flags = []
        if s["L0"]["mean"] <= floor + 5:
            flags.append("FLOOR")
        if s["L0"]["parse_fails"] + s["L1"]["parse_fails"] > args.n:
            flags.append("PARSE")
        if s["L0"]["reach"] < 0.05:
            flags.append("SIGNAL-DEAD")
        # the key honesty check: does the CI straddle zero?
        if ci == ci and abs(drop) < ci:
            flags.append("NS(CI crosses 0)")
        model = spec.split(":", 1)[-1] if ":" in spec else spec
        print(f"{model:22s} {s['L0']['mean']:9.2f} {s['L1']['mean']:9.2f} {drop:+8.2f} "
              f"{'+/-'+format(ci,'.1f'):>13s} {p:6.3f} "
              f"{s['L0']['parse_fails']:>2}/{s['L1']['parse_fails']:<2} "
              f"{s['L0']['reach']*100:5.0f}%  {' '.join(flags) if flags else 'ok'}")
        results.append((model, s, drop, ci, p, flags))

    print("-" * 88)
    print("\nRead-out:")
    informative = [(m, d, ci, p, f) for m, s, d, ci, p, f in results
                   if "FLOOR" not in f]
    if not informative:
        print("  Every rung sat on the floor -- none could defend at all. The pilot")
        print("  can't speak to retrieval yet: check the prompt, or the models are")
        print("  too small to act in this environment.")
        print("\n  (PILOT -- pre-register P1-P4 only after the full grid calibrates N.)")
        return 0

    # A drop counts as REAL only if it is negative AND significant (CI clears 0).
    sig_drops = [(m, d) for m, d, ci, p, f in informative
                 if d < 0 and "NS(CI crosses 0)" not in f and "PARSE" not in f]
    sig_any = [(m, d) for m, d, ci, p, f in informative
               if "NS(CI crosses 0)" not in f and "PARSE" not in f]
    signs = {("-" if d < 0 else "+") for m, d, ci, p, f in informative}

    if len(signs) > 1 and not sig_any:
        print("  Rungs DISAGREE IN SIGN and none is significant. This is the")
        print("  signature of NOISE, not a retrieval horizon -- at this N the")
        print("  effect (if any) is smaller than the sampling error.")
        print("  -> Increase N and re-run before drawing ANY conclusion.")
        print("     sd is ~40+ per cell; you likely need N>=40-100 per condition.")
    elif sig_drops:
        print("  SIGNIFICANT drop (CI clears zero) at:")
        for m, d in sig_drops:
            print(f"    {m}: {d:+.2f}")
        print("  If these are not confined to the very smallest rung, the effect")
        print("  looks real -> proceed to the full grid (Phase 3).")
    elif sig_any:
        print("  A significant effect exists but is not a clean drop everywhere:")
        for m, d, ci, p, f in informative:
            tag = "" if "NS(CI crosses 0)" in f else "  <- significant"
            print(f"    {m}: {d:+.2f} (95% CI +/-{ci:.1f}){tag}")
        print("  Read the pattern across rungs, not any single cell. A rung")
        print("  defending BETTER on the twin is worth understanding, not")
        print("  dismissing (e.g. it recognises the renamed role and reacts).")
    else:
        print("  No drop at any informative rung clears its confidence interval.")
        print("  Tentative NULL: contamination isn't measurably inflating these")
        print("  models here. Confirm with larger N before concluding.")

    print("\n  (PILOT -- pre-register P1-P4 only after the full grid calibrates N.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
