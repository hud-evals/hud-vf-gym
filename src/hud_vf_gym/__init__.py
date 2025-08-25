"""MCP-based HUD Gym environment for verifiers."""

import json

from datasets import Dataset, load_dataset

from .hud_vf_gym import HUDGym
from .rubrics import HUDBaseRubric


def load_environment(
    taskset: str,
    config_path: str,
    num_tasks: int | None = None,
    split: str = "train",
    **kwargs,
) -> HUDGym:
    """Load HUDGym environment from a HuggingFace dataset.

    Args:
        taskset: HuggingFace dataset identifier (required)
        config_path: Path to config file (required)
        num_tasks: Optional limit on number of tasks to load
        split: Dataset split to load (default: train)
        **kwargs: Additional arguments passed to HUDGym

    Returns:
        HUDGym: Configured environment
    """
    # Load HuggingFace dataset
    hf_dataset: Dataset = load_dataset(taskset, split=split)  # type: ignore

    if num_tasks is not None:
        hf_dataset = hf_dataset.select(range(num_tasks))

    # Workaround: Duplicate dataset 4x if it has fewer than 4 samples
    # This fixes a GRPO trainer initialization issue with small datasets
    # if len(hf_dataset) < 4:
    #     from datasets import concatenate_datasets
    #     hf_dataset = concatenate_datasets([hf_dataset] * (4 // len(hf_dataset) + 1))
    #     hf_dataset = hf_dataset.select(range(4))  # Ensure exactly 4 samples minimum

    # Create dataset for verifiers
    dataset = Dataset.from_dict(
        {
            "question": hf_dataset["prompt"],
            "task": [hf_dataset[i].get("id", f"task_{i}") for i in range(len(hf_dataset))],
            "answer": [
                hf_dataset[i].get("metadata", {}).get("answer", "")
                if isinstance(hf_dataset[i].get("metadata"), dict)
                else ""
                for i in range(len(hf_dataset))
            ],
            "info": [
                {
                    "mcp_config": hf_dataset[i]["mcp_config"]
                    if isinstance(hf_dataset[i]["mcp_config"], str)
                    else json.dumps(hf_dataset[i]["mcp_config"]),
                    "setup_tool": hf_dataset[i].get("setup_tool")
                    if isinstance(hf_dataset[i].get("setup_tool"), str)
                    else json.dumps(hf_dataset[i].get("setup_tool"))
                    if hf_dataset[i].get("setup_tool")
                    else None,
                    "evaluate_tool": hf_dataset[i].get("evaluate_tool")
                    if isinstance(hf_dataset[i].get("evaluate_tool"), str)
                    else json.dumps(hf_dataset[i].get("evaluate_tool"))
                    if hf_dataset[i].get("evaluate_tool")
                    else None,
                    "metadata": hf_dataset[i].get("metadata")
                    if isinstance(hf_dataset[i].get("metadata"), str)
                    else json.dumps(hf_dataset[i].get("metadata", {})),
                }
                for i in range(len(hf_dataset))
            ],
        }
    )

    return HUDGym(dataset=dataset, config_path=config_path, **kwargs)


__version__ = "0.1.0"

__all__ = [
    "HUDGym",
    "load_environment",
    "HUDBaseRubric",
]
