"""Generate a minimal HUD VF Gym configuration template.

This utility writes a single YAML template with placeholders and comments,
without any presets or assumptions.

Usage (CLI):
  python -m hud_vf_gym.config_template --output ./configs/template.yaml
  hudvf-config-init --output ./configs/template.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final


TEMPLATE: Final[str] = """# HUD VF Gym configuration template

# Job configuration (used for HUD telemetry)
job:
  name: "HUDGym Run"
  # Optional metadata to help identify runs
  metadata:
    dataset: "FILL_ME"      # e.g., "hud-evals/2048-taskset" or a local tag
    experiment: "baseline"  # e.g., "training", "ablation-1"
  # Optional dataset link for HUD UI (e.g., a Hugging Face URL)
  # dataset_link: "FILL_ME"

# System prompt used by the agent
system_prompt: |
  # TODO: Replace this block with your system prompt.
  # It can span multiple lines and will be passed to the agent.
  You are an AI assistant.

# Default settings
defaults:
  max_turns: 30

# Rubric configuration (optional)
rubric:
  weights:
    task_completion: 0.8  # Primary task completion weight
    tool_execution: 0.2   # Successful tool execution rate

# Restrict which tools the agent may call.
# Populate with tool names (e.g., ["computer"]).
# Leave as [] if you plan to fill later.
allowed_tools: []
"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a HUD VF Gym config template")
    p.add_argument("--output", required=True, help="Output YAML path for the template")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE, encoding="utf-8")
    print(f"Wrote template to {path}")


if __name__ == "__main__":
    main()
