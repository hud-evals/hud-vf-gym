"""Rubrics for MCP HUD Gym environment."""

from verifiers import Rubric
from verifiers.parsers.xml_parser import XMLParser


class HUDBaseRubric(Rubric):
    """Base rubric for evaluating HUD environment tasks."""

    def __init__(self, parser: XMLParser, weights: dict[str, float] | None = None):
        default_weights = {
            "task_completion": 0.8,
            "tool_execution": 0.1,
            "format_compliance": 0.1,
        }

        if weights:
            default_weights.update(weights)

        funcs = [
            self.hud_task_reward_func,  # Primary reward from HUD evaluation
            self.tool_execution_reward_func,  # Reward for successful tool calls
            parser.get_format_reward_func(),  # Reward for proper XML format and action syntax
        ]

        weights_list = [
            default_weights["task_completion"],
            default_weights["tool_execution"],
            default_weights["format_compliance"],
        ]

        super().__init__(funcs=funcs, weights=weights_list, parser=parser)
        self.parser = parser

    def hud_task_reward_func(self, completion: list[dict[str, str]], **kwargs) -> float:
        """Extract HUD task reward from state."""
        state = kwargs.get("state", {})
        return state.get("reward", 0.0)

    def tool_execution_reward_func(self, completion: list[dict[str, str]], **kwargs) -> float:
        """
        Reward function that checks tool execution success rate.

        Uses state tracking from HUDGym.
        """
        state = kwargs.get("state", {})
        tool_attempts = state.get("tool_attempts", 0)
        tool_successes = state.get("tool_successes", 0)

        if tool_attempts == 0:
            return 0.0

        return tool_successes / tool_attempts
