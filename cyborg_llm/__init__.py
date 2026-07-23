"""
cyborg_llm -- LLM agent harness for CybORG / CAGE Challenge 2.

Part of the Retrieval Horizon study (Phase 0).

This file must exist for `from cyborg_llm.env import ...` to work.
It may be empty; the contents below are convenience only.

Quick start:
    from cyborg_llm.providers import get_provider
    from cyborg_llm.runner import run_episode

    p = get_provider("mock")                    # no API key, no cost
    r = run_episode(p, seed=0, level="L0")
    print(r["total_reward"])

Verified environment (Phase 0 recon, July 2026):
    Python 3.12, CybORG 2.1, CAGE Challenge 2.
    Baseline band: 0.00 (no attack) to -211.78 (passive blue vs B_line).
    Seeding: CybORG 2.1 has NO seed API -- see env.make_env().
"""

__version__ = "0.1.0"

# Pinned for reproducibility -- log these with every run.
CYBORG_VERSION = "2.1"
SCENARIO = "Scenario2.yaml"

from .env import (  # noqa: F401
    make_env,
    serialize_obs,
    action_menu,
    parse_action,
    default_scenario_path,
    ALL_ACTION_NAMES,
)
from .providers import get_provider  # noqa: F401
from .runner import (  # noqa: F401
    run_episode,
    append_record,
    completed_keys,
    MAX_TURNS,
)

__all__ = [
    "make_env", "serialize_obs", "action_menu", "parse_action",
    "default_scenario_path", "ALL_ACTION_NAMES",
    "get_provider",
    "run_episode", "append_record", "completed_keys", "MAX_TURNS",
    "CYBORG_VERSION", "SCENARIO", "__version__",
]
