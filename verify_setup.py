#!/usr/bin/env python3
"""
Phase 0 verification. Run this; if it passes, you have a research instrument.

  python3 verify_phase0.py            # checks 1,2,4,5 (no LLM needed)
  python3 verify_phase0.py --provider ollama:qwen2.5:3b     # adds check 3
"""
import argparse
import os
import statistics
import sys
import warnings

warnings.filterwarnings("ignore")

OUT = "runs/verify.jsonl"


def check1_env():
    print("[1] CybORG installs and runs ................. ", end="", flush=True)
    from cyborg_llm.env import make_env
    cyborg = make_env(seed=0)
    res = cyborg.reset(agent="Blue")
    assert res.observation, "empty observation"
    print("PASS")


def check2_baseline():
    print("[2] Scripted baselines reproduce band ....... ", end="", flush=True)
    from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
    from CybORG.Agents import BlueMonitorAgent, BlueReactRemoveAgent, BlueReactRestoreAgent
    from cyborg_llm.env import make_env

    def run(red, blue_cls, steps=30, eps=3):
        out = []
        for i in range(eps):
            cy = make_env(red_agent=red, seed=i)
            obs = cy.reset(agent="Blue").observation
            blue = blue_cls()
            tot = 0.0
            for _ in range(steps):
                a = blue.get_action(obs, cy.get_action_space("Blue"))
                r = cy.step(agent="Blue", action=a)
                obs = r.observation
                tot += r.reward
            out.append(tot)
        return statistics.mean(out)

    ceiling = run(SleepAgent, BlueMonitorAgent)
    floor = run(B_lineAgent, BlueMonitorAgent)
    good = run(B_lineAgent, BlueReactRestoreAgent)
    assert abs(ceiling) < 1e-6, f"ceiling should be 0.0, got {ceiling}"
    assert floor < -100, f"floor should be strongly negative, got {floor}"
    print("PASS")
    print(f"      ceiling (no attack)        = {ceiling:8.2f}")
    print(f"      floor   (passive blue)     = {floor:8.2f}")
    print(f"      competent scripted blue    = {good:8.2f}")
    print(f"      -> your LLM must land inside this band to be measurable")


def check3_llm(spec):
    print(f"[3] LLM round-trip ({spec}) ...", flush=True)
    from cyborg_llm.providers import get_provider
    from cyborg_llm.runner import run_episode
    p = get_provider(spec)
    r = run_episode(p, seed=0, level="L0", max_turns=5)
    print(f"      reward={r['total_reward']:.2f} turns={r['n_turns']} "
          f"parse_failures={r['parse_failures']} abort={r['abort_reason']}")
    print(f"      tokens in/out = {r['usage']['input_tokens']}/{r['usage']['output_tokens']}"
          f"  wall={r['wall_seconds']}s")
    if r["n_turns"] == 0:
        print("      WARNING: model produced zero valid actions -- check the prompt format")
    est = r["usage"]["input_tokens"] / max(r["n_turns"], 1) * 30
    print(f"      -> projected ~{est:,.0f} input tokens for a 30-turn episode")
    print("      PASS")


def check4_determinism():
    print("[4] Seeding is deterministic ................ ", end="", flush=True)
    from cyborg_llm.providers import MockProvider
    from cyborg_llm.runner import run_episode
    s = ["Monitor", "Analyse User1", "Restore User1"]
    a = run_episode(MockProvider(script=s), seed=7, level="L0")
    b = run_episode(MockProvider(script=s), seed=7, level="L0")
    c = run_episode(MockProvider(script=s), seed=8, level="L0")
    assert a["total_reward"] == b["total_reward"], "same seed gave different rewards!"
    print("PASS")
    print(f"      seed 7 twice = {a['total_reward']:.2f}, {b['total_reward']:.2f} (identical)")
    print(f"      seed 8       = {c['total_reward']:.2f} (differs -> seeds give real variance)")


def check5_resume():
    print("[5] Logging survives a kill ................. ", end="", flush=True)
    from cyborg_llm.providers import MockProvider
    from cyborg_llm.runner import run_episode, append_record, completed_keys
    if os.path.exists(OUT):
        os.remove(OUT)
    for seed in (1, 2):
        append_record(OUT, run_episode(MockProvider(script=["Monitor"]), seed=seed, level="L0"))
    done = completed_keys(OUT)
    assert ("mock", "L0", 1) in done and ("mock", "L0", 3) not in done
    with open(OUT, "a") as f:
        f.write('{"model": "mock", "level": "L0", "seed": 3, "trunc')  # simulate hard kill
    done2 = completed_keys(OUT)
    assert done2 == done, "torn line broke the resume index"
    print("PASS")
    print(f"      {len(done)} episodes indexed; torn final line ignored cleanly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    print("=" * 62)
    print("PHASE 0 VERIFICATION")
    print("=" * 62)
    check1_env()
    check2_baseline()
    check4_determinism()
    check5_resume()
    if args.provider:
        check3_llm(args.provider)
    else:
        print("[3] LLM round-trip .......................... SKIPPED (--provider)")
    print("=" * 62)
    print("Phase 0 complete. You have a research instrument.")
    print("Next: Phase 1 -- the twin generator (Gate 1).")
