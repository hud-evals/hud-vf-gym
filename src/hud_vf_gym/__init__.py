"""MCP-based HUD Gym environment for verifiers."""

import json
import os

from . import _patches  # noqa: F401

from datasets import Dataset, load_dataset

from .hud_vf_gym import HUDGym
from .rubrics import HUDBaseRubric


def load_environment(
    taskset: str,
    config_path: str,
    num_tasks: int | None = None,
    split: str = "train",
    replicate_to: int | None = None,
    task_index: int | None = None,
    **kwargs,
) -> HUDGym:
    """Load HUDGym environment from a HuggingFace dataset or JSON file.

    Args:
        taskset: HuggingFace dataset identifier or local JSON/JSONL path
        config_path: Path to config file (required)
        num_tasks: Optional limit on number of tasks to load
        split: Dataset split to load (default: train)
        **kwargs: Additional arguments passed to HUDGym

    Returns:
        HUDGym: Configured environment
    """
    assert config_path is not None, "config_path is required"

    def _normalize_examples(examples: list[dict]) -> Dataset:
        # Create dataset for verifiers from a list of task dicts
        prompts = [ex.get("prompt", "") for ex in examples]
        tasks = [ex.get("id", f"task_{i}") for i, ex in enumerate(examples)]
        answers = []
        infos = []
        for ex in examples:
            meta = ex.get("metadata", {})
            # If metadata is a JSON string, load to extract answer if present
            if isinstance(meta, str):
                try:
                    meta_obj = json.loads(meta)
                except Exception:
                    meta_obj = {}
            else:
                meta_obj = meta if isinstance(meta, dict) else {}
            answers.append(meta_obj.get("answer", ""))
            infos.append(
                {
                    "mcp_config": ex["mcp_config"]
                    if isinstance(ex.get("mcp_config"), str)
                    else json.dumps(ex.get("mcp_config", {})),
                    "setup_tool": ex.get("setup_tool")
                    if isinstance(ex.get("setup_tool"), str)
                    else json.dumps(ex.get("setup_tool"))
                    if ex.get("setup_tool") is not None
                    else None,
                    "evaluate_tool": ex.get("evaluate_tool")
                    if isinstance(ex.get("evaluate_tool"), str)
                    else json.dumps(ex.get("evaluate_tool"))
                    if ex.get("evaluate_tool") is not None
                    else None,
                    "metadata": meta if isinstance(meta, str) else json.dumps(meta_obj),
                }
            )

        return Dataset.from_dict({"question": prompts, "task": tasks, "answer": answers, "info": infos})

    # If caller provided a local path in taskset, treat it as JSON input
    if isinstance(taskset, str) and os.path.exists(taskset):
        # Load from JSON or JSONL file
        with open(taskset, "r") as f:
            raw = f.read()

        examples: list[dict]
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict) and "data" in loaded and isinstance(loaded["data"], list):
                examples = loaded["data"]
            elif isinstance(loaded, list):
                examples = loaded
            else:
                raise ValueError("Unsupported JSON structure; expected list or {data: list}")
        except json.JSONDecodeError:
            # Try JSON Lines
            examples = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        examples.append(obj)
                except Exception:
                    continue

            if not examples:
                raise

        # If a single task index is specified, select that one task (wrap around if out of bounds)
        if task_index is not None and len(examples) > 0:
            idx = task_index % len(examples)
            examples = [examples[idx]]
        elif num_tasks is not None:
            examples = examples[:num_tasks]

        # Optionally replicate to reach a desired count
        target_count = replicate_to or (num_tasks if num_tasks is not None else len(examples))
        if target_count > len(examples) and len(examples) > 0:
            base = examples
            idx = 0
            while len(examples) < target_count:
                ex = dict(base[idx % len(base)])
                base_id = ex.get("id", f"task_{idx % len(base)}")
                ex["id"] = f"{base_id}__dup{len(examples)}"
                examples.append(ex)
                idx += 1

        dataset = _normalize_examples(examples)
        return HUDGym(dataset=dataset, config_path=config_path, **kwargs)

    # Fallback to HuggingFace dataset path
    assert taskset is not None, "taskset must be a HF dataset ID or a local JSON/JSONL path"

    hf_dataset: Dataset = load_dataset(taskset, split=split)  # type: ignore

    # Only truncate if requested size is less than the dataset length.
    if num_tasks is not None:
        try:
            total_len = len(hf_dataset)  # type: ignore[arg-type]
        except Exception:
            total_len = None  # fallback if not supported
        if total_len is None or num_tasks < total_len:
            hf_dataset = hf_dataset.select(range(num_tasks))

    # If a single task index is specified, select that one task (wrap around if out of bounds)
    if task_index is not None:
        try:
            total_len = len(hf_dataset)  # type: ignore[arg-type]
        except Exception:
            total_len = 0
        if total_len == 0:
            examples = []
        else:
            idx = task_index % total_len
            ex_row = hf_dataset[idx]
            examples = [
                {
                    "id": ex_row.get("id", f"task_{idx}"),
                    "prompt": ex_row.get("prompt", ""),
                    "mcp_config": ex_row.get("mcp_config"),
                    "setup_tool": ex_row.get("setup_tool"),
                    "evaluate_tool": ex_row.get("evaluate_tool"),
                    "metadata": ex_row.get("metadata", {}),
                }
            ]
    else:
        examples = [
            {
                "id": hf_dataset[i].get("id", f"task_{i}"),
                "prompt": hf_dataset[i].get("prompt", ""),
                "mcp_config": hf_dataset[i].get("mcp_config"),
                "setup_tool": hf_dataset[i].get("setup_tool"),
                "evaluate_tool": hf_dataset[i].get("evaluate_tool"),
                "metadata": hf_dataset[i].get("metadata", {}),
            }
            for i in range(len(hf_dataset))
        ]

    # Optionally replicate to reach a desired count
    target_count = replicate_to or (num_tasks if num_tasks is not None else len(examples))
    if target_count > len(examples) and len(examples) > 0:
        base = examples.copy()
        idx = 0
        while len(examples) < target_count:
            ex = dict(base[idx % len(base)])
            base_id = ex.get("id", f"task_{idx % len(base)}")
            ex["id"] = f"{base_id}__dup{len(examples)}"
            examples.append(ex)
            idx += 1

    dataset = _normalize_examples(examples)

    return HUDGym(dataset=dataset, config_path=config_path, **kwargs)


__version__ = "0.1.2"

__all__ = [
    "HUDGym",
    "load_environment",
    "HUDBaseRubric",
]
