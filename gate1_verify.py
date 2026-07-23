#!/usr/bin/env python3
"""
GATE 1 -- can we build difficulty-matched twins with a LIVE retrieval signal?

Two conditions, both required. Either alone is worthless.

  (1) TWINS MATCHED.  A name-blind agent must score the same on L0 and the twin.
      Then any later LLM drop cannot be difficulty -- only names changed.

  (2) SIGNAL LIVE.    Red must actually threaten the crown jewel within the
      episode. Otherwise "Op_Server0 is critical" is irrelevant, renaming it
      changes nothing, and a null result would mean nothing.

Getting BOTH required fixing CAGE-2 at three levels (see probes.py):
  blue agent : hardcoded `!= 'User0'`
  red agents : hardcoded 'Op_Server0'; tier-by-substring 'Op'/'Ent'
  SIMULATOR  : `if 'Enterprise' in session.host` gating Operational access
and switching red from Meander (crown jewel reached 0/60 in 30 steps) to
B_line (~12-17%).

Usage:  python3 gate1_verify.py        # ~4 min, no API keys, $0
"""
import os
import statistics
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

import networkx as nx
from scipy import stats

from cyborg_llm.env import default_scenario_path
from cyborg_llm.twins import (load_scenario, make_twin, write_scenario,
                              host_graph, path_profile, red_start, crown_jewel)
from cyborg_llm.probes import (make_probe, make_bline_probe_class,
                               patch_simulator_enterprise_check)

N_EPISODES = 200      # episodes cost ~0.16s; sd~42 with B_line, so buy power
STEPS = 30            # the protocol horizon
TWIN_SPECS = [("L1", "alt_a"), ("L1", "alt_b"), ("L1", "neutral")]
# L2 REMOVED: moving red's entry point changes difficulty even with canonical
# names (-12.69 -> -3.00). User0 is structurally privileged. Rejected, not
# adjusted.

NODE_MATCH = lambda a, b: (a["image"] == b["image"]
                           and a["conf"] == b["conf"]
                           and a["avail"] == b["avail"])


def run(scenario, path, n=N_EPISODES, steps=STEPS):
    """Name-blind blue + red probes. Returns (rewards, crown_jewel_reach_rate)."""
    import random
    import numpy as np
    from CybORG import CybORG

    patch_simulator_enterprise_check(scenario)   # global; re-apply per scenario
    goal = crown_jewel(scenario)
    out, reached = [], 0
    for seed in range(n):
        random.seed(seed)
        np.random.seed(seed)
        cy = CybORG(path, "sim", agents={"Red": make_bline_probe_class(scenario)})
        obs = cy.reset(agent="Blue").observation
        blue = make_probe(scenario, restore=True)
        tot, saw = 0.0, False
        for _ in range(steps):
            a = blue.get_action(obs, cy.get_action_space("Blue"))
            r = cy.step(agent="Blue", action=a)
            obs = r.observation
            tot += r.reward
            if goal in str(cy.environment_controller.action.get("Red")):
                saw = True
        out.append(tot)
        reached += saw
    return out, reached / n


def main():
    orig = load_scenario(default_scenario_path())
    prof0, g0 = path_profile(orig), host_graph(orig)

    print("=" * 74)
    print("GATE 1 -- DIFFICULTY-MATCHED TWINS WITH A LIVE RETRIEVAL SIGNAL")
    print("=" * 74)
    print(f"\nL0 canonical: red_start={red_start(orig)}  goal={crown_jewel(orig)}")
    print(f"  hosts={prof0['n_hosts']} edges={prof0['n_edges']} "
          f"hops_to_goal={prof0['hops_start_to_goal']}")

    tmp = tempfile.mkdtemp(prefix="twins_")
    l0 = os.path.join(tmp, "L0.yaml")
    write_scenario(orig, l0)

    print(f"\nB_line red + name-blind blue, simulator patched. N={N_EPISODES} x {STEPS} steps")
    base, base_reach = run(orig, l0)
    print(f"  L0: mean={statistics.mean(base):8.2f} sd={statistics.stdev(base):6.2f} "
          f"| red reached crown jewel in {base_reach*100:.1f}% of episodes")

    if base_reach < 0.05:
        print("\n  *** SIGNAL DEAD: red barely threatens the crown jewel. ***")
        print("  Renaming it cannot matter. Do NOT proceed to Gate 2.")
        return 1

    print("\n" + "-" * 74)
    print(f"{'TWIN':14s} {'ISO':5s} {'PROF':5s} {'MEAN':>9s} {'DELTA':>8s} "
          f"{'p':>7s} {'GOAL%':>7s}  VERDICT")
    print("-" * 74)

    ok = True
    for level, scheme in TWIN_SPECS:
        twin, _ = make_twin(orig, level=level, scheme=scheme)
        tag = f"{level}_{scheme}"
        tp = os.path.join(tmp, f"{tag}.yaml")
        write_scenario(twin, tp)

        iso = nx.is_isomorphic(g0, host_graph(twin), node_match=NODE_MATCH)
        prof_ok = path_profile(twin) == prof0
        rew, reach = run(twin, tp)
        _, p = stats.ttest_ind(base, rew, equal_var=False)

        matched, live = p > 0.05, reach >= 0.05
        good = iso and prof_ok and matched and live
        ok &= good
        print(f"{tag:14s} {str(iso):5s} {str(prof_ok):5s} {statistics.mean(rew):9.2f} "
              f"{statistics.mean(rew)-statistics.mean(base):+8.2f} {p:7.3f} "
              f"{reach*100:6.1f}%  {'PASS' if good else 'FAIL'}")

    print("-" * 74)
    print("\nInterpretation:")
    print("  Probes are name-blind BY CONSTRUCTION, and the simulator's")
    print("  `'Enterprise' in session.host` gate is patched to use structural")
    print("  tier membership. The shipped CAGE-2 stack is NOT name-blind at any")
    print("  level -- agents or environment.")
    print("  MATCHED + LIVE => a later LLM drop cannot be difficulty, and the")
    print("  thing being disguised actually matters.")
    print()
    if ok:
        print("GATE 1: PASS -- twins difficulty-matched AND signal live.")
        print("Next: Gate 2 -- does the drop exist at a rung anyone deploys?")
        return 0
    print("GATE 1: FAIL. Reject the twin; never adjust it to pass.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
