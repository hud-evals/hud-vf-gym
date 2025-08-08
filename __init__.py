"""HUD Gym v2 - Environment for HUD tasks with verifiers."""

from datasets import Dataset, load_dataset
from hud.datasets import to_taskconfigs

from .hud_vf_gym import HUDGym
from .rubrics import HUDEvaluationRubric


def load_environment(
    taskset: str = "hud-evals/gmail-taskset",
    num_tasks: int | None = None,
    split: str = "train",
    **kwargs,
) -> HUDGym:
    """Load HUDGym environment from a HuggingFace dataset.

    Args:
        taskset: HuggingFace dataset identifier (default: hud-evals/gmail-taskset)
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

    # Convert to TaskConfigs to extract HUD-specific fields
    task_configs = to_taskconfigs(hf_dataset)

    # Create dataset with proper structure for verifiers
    dataset = Dataset.from_dict(
        {
            "question": [task.prompt for task in task_configs],
            "task": [task.id or f"task_{i}" for i, task in enumerate(task_configs)],
            "answer": [task.metadata.get("answer", "") for task in task_configs],
            "info": [
                {
                    "mcp_config": task.mcp_config,
                    "setup_tool": task.setup_tool.model_dump() if task.setup_tool else None,
                    "evaluate_tool": task.evaluate_tool.model_dump() if task.evaluate_tool else None,
                    "metadata": task.metadata,
                }
                for task in task_configs
            ],
        }
    )

    return HUDGym(dataset=dataset, **kwargs)


__version__ = "2.0.0"

__all__ = [
    "HUDGym",
    "HUDEvaluationRubric",
    "load_environment",
]
