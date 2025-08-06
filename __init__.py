"""MCP-based HUD Gym environment for verifiers."""

import json

from datasets import Dataset, load_dataset
from hud.datasets import to_taskconfigs

from .hud_vf_gym import HUDGym
from .rubrics import HUDToolRubric


def load_environment(
    source: str,
    config_path: str | None = None,
    num_tasks: int | None = None,
    **kwargs,
) -> HUDGym:
    """Load HUDGym environment from a taskset file or HuggingFace dataset.

    Args:
        source: Either a path to taskset JSON file or HuggingFace dataset identifier
        config_path: Optional path to config file
        num_tasks: Optional limit on number of tasks to load
        **kwargs: Additional arguments passed to HUDGym

    Returns:
        HUDGym: Configured environment
    """
    # Check if source is a file path (taskset.json)
    if source.endswith('.json'):
        # Load taskset
        with open(source) as f:
            taskset_data = json.load(f)

        # Get tasks
        task_list = taskset_data.get("tasks", [])
        if num_tasks is not None:
            task_list = task_list[:num_tasks]

        dataset = Dataset.from_dict({
            "question": [t.get("prompt", "") for t in task_list],
            "task": [t.get("metadata", {}).get("id", f"task_{i}") for i, t in enumerate(task_list)],
            "answer": [t.get("metadata", {}).get("answer", "") for t in task_list],
            "info": [{
                "mcp_config": t.get("mcp_config"),
                "setup_tool": t.get("setup_tool"),
                "evaluate_tool": t.get("evaluate_tool"),
                "metadata": t.get("metadata", {})
            } for t in task_list]
        })
    else:
        #TODO: Fix to_taskconfigs in hud_sdk
        dataset_name = source
        split = "train"

        # Load HuggingFace dataset
        hf_dataset: Dataset = load_dataset(dataset_name, split=split) # type: ignore

        # Convert to TaskConfigs to extract HUD-specific fields
        task_dataset = to_taskconfigs(hf_dataset)

        # Create dataset with proper structure
        dataset = Dataset.from_dict({
            "question": [row["task"].prompt for row in task_dataset],
            "task": [row["task"].id or f"task_{i}" for i, row in enumerate(task_dataset)],
            "answer": [row["task"].metadata.get("answer", "") for row in task_dataset],
            "info": [{
                "mcp_config": row["task"].mcp_config,
                "setup_tool": row["task"].setup_tool,
                "evaluate_tool": row["task"].evaluate_tool,
                "metadata": row["task"].metadata
            } for row in task_dataset]
        })

    return HUDGym(dataset=dataset, config_path=config_path, **kwargs)


__version__ = "0.1.0"

__all__ = [
    "HUDGym",
    "load_environment",
    "HUDToolRubric",
]
