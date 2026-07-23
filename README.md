# Surface-Form Leakage in Autonomous Cyber-Defense Benchmarks

Code and data accompanying the paper:

> **Surface-Form Leakage in Autonomous Cyber-Defense Benchmarks: A Case Study and a Difficulty-Matched Control**
> U. Kalaiah. *International Journal on Advanced Science, Engineering and Information Technology*, 20XX.
> DOI: *(add on acceptance)*

This repository contains everything needed to reproduce the tables and figures in the paper: the difficulty-matched twin generator, the name-blind scripted probes and the simulator correction, the LLM agent harness, the logged episodes, and the analysis scripts.

---

## What this code does

The paper asks whether LLM agents evaluated on CAGE-2 are reasoning about network structure or recognizing benchmark-specific identifiers, and reports three findings:

1. **CAGE-2 encodes canonical hostnames in its transition logic.** Subnet-access permission is gated by a substring test on host names, so naive renaming silently changes attacker reachability. `cyborg_llm/probes.py` documents the dependency and provides a structural correction.
2. **A difficulty-matched twin control is constructible and verifiable.** `cyborg_llm/twins.py` generates renamed scenarios; `gate1_verify.py` verifies that a name-blind agent scores identically on original and twin.
3. **Benchmark reward is confounded by agent play-style.** `gate2_pilot.py` runs the multi-model case study; `reanalyze.py` evaluates it under several metrics.

---

## Installation

Requires Python 3.9 or newer.

```bash
git clone <REPO-URL>
cd surface-form-leakage-cage2
pip install -r requirements.txt
```

### CybORG dependency

CAGE-2 is not bundled here (it is third-party code with its own license). Install it separately:

```bash
git clone --depth 1 https://github.com/cage-challenge/cage-challenge-2.git
cd cage-challenge-2/CybORG
python3 -m pip install -e .
cd ../..
```

Verify the installation:

```bash
python3 verify_setup.py
```

### For the LLM experiments only

The multi-model case study requires [Ollama](https://ollama.com) and the Qwen2.5 models:

```bash
ollama pull qwen2.5:0.5b qwen2.5:1.5b qwen2.5:3b qwen2.5:7b qwen2.5:14b
```

The difficulty-verification results (Table I) require **no models and no API keys**.

---

## Reproducing the paper

| Paper artifact | Command | Runtime |
|---|---|---|
| **Table I** — difficulty verification | `python3 gate1_verify.py` | ~4 min, no models |
| **Table II** — multi-model case study | `python3 gate2_pilot.py --rungs ollama:qwen2.5:0.5b ollama:qwen2.5:1.5b ollama:qwen2.5:3b ollama:qwen2.5:7b ollama:qwen2.5:14b --n 40` | hours (LLM-bound) |
| **Table II**, 3B at N=150 | `python3 gate2_pilot.py --rungs ollama:qwen2.5:3b --n 150` | ~1 hr |
| Confidence intervals | `python3 compute_ci.py` | seconds |
| Multi-metric reanalysis | `python3 reanalyze.py` | seconds |
| **Fig. 1** — topology | `python3 figures/fig1_topology.py` | seconds |
| **Fig. 2** — action distribution | `python3 figures/fig2_actionmix.py` | seconds |

`gate2_pilot.py` is resumable: it logs to `runs/gate2.jsonl` and skips episodes already present, so it can be interrupted and restarted freely.

To reproduce the analysis without re-running the models, use the logged episodes shipped in `data/`:

```bash
mkdir -p runs && cp data/gate2.jsonl runs/
python3 compute_ci.py
python3 reanalyze.py
```

---

## Determinism

CybORG 2.1 exposes no seeding interface. Determinism is obtained by seeding the global `random` and `numpy` generators **before** environment instantiation, as done throughout this code. With this procedure the per-episode rewards reproduce exactly — to the last decimal place — across operating systems, Python versions, and processor architectures. Results in the paper were produced on macOS (Apple silicon, Python 3.9) and independently reproduced on x86-64 Linux (Python 3.12).

---

## Repository structure

```
cyborg_llm/
  env.py           observation serializer and action parser
  providers.py     model backends (Ollama, mock, and others)
  runner.py        episode loop, guardrails, resumable JSONL logging
  twins.py         rename schemes, directed-graph difficulty verification
  probes.py        name-blind scripted agents + simulator correction
verify_setup.py    environment and dependency checks
gate1_verify.py    difficulty verification (Table I)
gate2_pilot.py     multi-model case study (Table II)
compute_ci.py      confidence intervals and significance tests
reanalyze.py       multi-metric reanalysis (reward, reach, survival)
data/
  gate2.jsonl      logged episodes underlying Table II and Fig. 2
figures/
  fig1_topology.py, fig2_actionmix.py
```

---

## Attribution

`cyborg_llm/probes.py` contains agent logic adapted from the CAGE-2 reference agents (`Meander.py`, `B_line.py`, `BlueReactAgent.py`), modified so that host identity is read from the scenario rather than hardcoded. Those portions remain © their original authors and are used under the terms of the CAGE-2 / CybORG license; see that project for details. The simulator correction described in the paper is applied at runtime and does not modify the installed CybORG package.

## License

*(add — e.g. MIT or Apache-2.0, subject to compatibility with the CybORG license)*

## Citation

```bibtex
@article{kalaiah20XX_surfaceform,
  title   = {Surface-Form Leakage in Autonomous Cyber-Defense Benchmarks:
             A Case Study and a Difficulty-Matched Control},
  author  = {Kalaiah, Umashankara},
  journal = {International Journal on Advanced Science, Engineering and
             Information Technology},
  year    = {20XX}
}
```
