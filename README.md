# Detecting and Correcting Surface-Form Leakage in Autonomous Cyber-Defense Benchmarks

Code and data accompanying the paper:

> **Detecting and Correcting Surface-Form Leakage in Autonomous Cyber-Defense Benchmarks**
> U. Kalaiah. *International Journal of Systems and Software Security and Protection (IJSSSP)*, IGI Global Scientific Publishing, 20XX.
> DOI: *(add on acceptance)*

This repository contains everything needed to reproduce the tables and figures in the paper: the difficulty-matched twin generator, the name-blind scripted probes and the simulator correction, the LLM agent harness, the logged episodes, and the analysis scripts.

---



## What's in this repository

This code implements the audit, control, and case study described in the paper. Briefly, by file:

- `cyborg_llm/probes.py` — the corrected subnet-access check, plus the name-blind scripted probe agents.
- `cyborg_llm/twins.py` — the three host-renaming schemes and the directed-graph isomorphism check.
- `gate1_verify.py` — runs the probes on the original network and each twin.
- `gate2_pilot.py` — runs the multi-model Qwen2.5 case study across original and renamed networks.
- `compute_ci.py` / `reanalyze.py` — statistics and the multi-metric breakdown used in the paper's tables.

See the paper for the methodology, findings, and their interpretation.

---



## Installation

Requires Python 3.9 or newer.

```bash
git clone <REPO-URL>
cd surface-form-leakage-cage2
python3 -m pip install -r requirements.txt
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

The difficulty-verification results (Table 1) require **no models and no API keys**.

---



## Reproducing the paper


| Paper artifact                        | Command                                                                                                                                | Runtime           |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| **Table 1** — difficulty verification | `python3 gate1_verify.py`                                                                                                              | ~4 min, no models |
| **Table 2** — multi-model case study  | `python3 gate2_pilot.py --rungs ollama:qwen2.5:0.5b ollama:qwen2.5:1.5b ollama:qwen2.5:3b ollama:qwen2.5:7b ollama:qwen2.5:14b --n 40` | hours (LLM-bound) |
| **Table 2**, 3B at N=150              | `python3 gate2_pilot.py --rungs ollama:qwen2.5:3b --n 150`                                                                             | ~1 hr             |
| Confidence intervals                  | `python3 compute_ci.py`                                                                                                                | seconds           |
| Multi-metric reanalysis               | `python3 reanalyze.py`                                                                                                                 | seconds           |
| **Figure 1** — topology               | `python3 figures/fig1_topology.py`                                                                                                     | seconds           |
| **Figure 2** — action distribution    | `python3 figures/fig2_actionmix.py`                                                                                                    | seconds           |


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
gate1_verify.py    difficulty verification (Table 1)
gate2_pilot.py     multi-model case study (Table 2)
compute_ci.py      confidence intervals and significance tests
reanalyze.py       multi-metric reanalysis (reward, reach, survival)
data/
  gate2.jsonl      logged episodes underlying Table 2 and Figure 2
figures/
  fig1_topology.py, fig2_actionmix.py
```

---



## Attribution

`cyborg_llm/probes.py` contains agent logic adapted from the CAGE-2 reference agents (`Meander.py`, `B_line.py`, `BlueReactAgent.py`), modified so that host identity is read from the scenario rather than hardcoded. Those portions remain © their original authors and are used under the terms of the CAGE-2 / CybORG license; see that project for details. The simulator correction described in the paper is applied at runtime and does not modify the installed CybORG package.

## License

MIT License. See [LICENSE](LICENSE) for the full text.

## Citation

```bibtex
@article{kalaiah20XX_surfaceform,
  title   = {Detecting and Correcting Surface-Form Leakage in Autonomous
             Cyber-Defense Benchmarks},
  author  = {Kalaiah, Umashankara},
  journal = {International Journal of Systems and Software Security and
             Protection (IJSSSP)},
  year    = {20XX}
}
```

