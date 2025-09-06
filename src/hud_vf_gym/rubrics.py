"""Rubrics for HUD Gym environment."""

from verifiers import Rubric


class HUDBaseRubric(Rubric):
    """Base rubric for evaluating HUD environment tasks."""

    def __init__(self, weights: dict[str, float] | None = None):
        default_weights = {
            "task_completion": 0.8,
            "tool_execution": 0.2,
        }

        if weights:
            default_weights.update(weights)

        funcs = [
            self.hud_task_reward_func,  # Primary reward from HUD evaluation
            self.tool_execution_reward_func,  # Reward for successful tool calls
        ]

        weights_list = [
            default_weights["task_completion"],
            default_weights["tool_execution"],
        ]

        super().__init__(funcs=funcs, weights=weights_list, parser=None)

    def hud_task_reward_func(self, completion: list[dict[str, str]], **kwargs) -> float:
        """Extract HUD task reward from state."""
        state = kwargs.get("state", {})
        # The reward comes from the agent's trace evaluation
        return state.get("reward", 0.0)

    def tool_execution_reward_func(self, completion: list[dict[str, str]], **kwargs) -> float:
        """
        Reward function that checks tool execution success rate.

        Uses trace from GenericOpenAIChatAgent to calculate success rate.
        """
        state = kwargs.get("state", {})
        trace = state.get("trace")

        if not trace or not trace.trace:  # trace.trace is the list of steps
            return 0.0

        tool_calls = 0
        successful_calls = 0

        # Count tool calls and successes from trace
        for step in trace.trace:  # trace.trace contains the steps
            if step.category == "mcp":
                tool_calls += 1
                # Check if the tool call was successful
                if hasattr(step.result, "isError") and not step.result.isError:
                    successful_calls += 1
                elif not hasattr(step.result, "isError"):
                    # If no error field, assume success
                    successful_calls += 1

        if tool_calls == 0:
            # No tools called, but that might be fine
            return 1.0

        return successful_calls / tool_calls
