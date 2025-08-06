"""Rubrics for MCP HUD Gym environment."""

from collections.abc import Callable

from verifiers import Rubric
from verifiers.parsers.xml_parser import XMLParser


class HUDToolRubric(Rubric):
    """Rubric for evaluating tool usage in HUD environment."""

    def __init__(self, parser: XMLParser, weights: dict[str, float] | None = None):
        default_weights = {
            "task_completion": 0.7,
            "tool_execution": 0.1,
            "format_compliance": 0.1,
            "screenshot_behavior": 0.05,
            "thinking_quality": 0.05,
        }

        if weights:
            default_weights.update(weights)

        funcs = [
            self.hud_task_reward_func,  # Primary reward from HUD evaluation
            self.tool_execution_reward_func,  # Reward for successful tool calls
            parser.get_format_reward_func(),  # Reward for proper XML format and action syntax
            self.get_screenshot_frequency_reward_func(),  # Reward for taking screenshot first
            self.get_thinking_quality_reward_func(),  # Reward for concise, quality thinking
        ]

        weights_list = [
            default_weights["task_completion"],
            default_weights["tool_execution"],
            default_weights["format_compliance"],
            default_weights["screenshot_behavior"],
            default_weights["thinking_quality"],
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

    def get_tool_diversity_reward_func(self) -> Callable:
        """
        Returns a reward function that encourages using diverse tools.
        """

        def tool_diversity_reward(completion: list[dict[str, str]], **kwargs) -> float:
            tools_used = set()

            for msg in completion:
                if msg["role"] == "assistant":
                    parsed = self.parser.parse(msg["content"])
                    if hasattr(parsed, "action") and parsed.action:
                        tools_used.add(parsed.action["name"])

            # Reward based on number of unique tools used
            # Cap at 5 for normalization
            return min(len(tools_used) / 5.0, 1.0)

        return tool_diversity_reward

    def get_screenshot_frequency_reward_func(self) -> Callable:
        """
        Returns a reward function that checks screenshot usage.

        Strongly rewards taking a screenshot as the first action.
        Allows but doesn't encourage additional screenshots.
        """

        def screenshot_reward(completion: list[dict[str, str]], **kwargs) -> float:
            first_action_screenshot = False
            explicit_screenshots = 0
            total_actions = 0

            for msg in completion:
                if msg["role"] == "assistant":
                    parsed = self.parser.parse(msg["content"])
                    if hasattr(parsed, "action") and parsed.action:
                        # Check if first action is screenshot
                        if total_actions == 0 and parsed.action["name"] == "screenshot":
                            first_action_screenshot = True

                        total_actions += 1
                        if parsed.action["name"] == "screenshot":
                            explicit_screenshots += 1

            if total_actions == 0:
                return 0.0

            first_action_score = 1.0 if first_action_screenshot else 0.0

            if explicit_screenshots <= 1:
                excess_penalty = 1.0
            elif explicit_screenshots == 2:
                excess_penalty = 0.8
            else:
                excess_penalty = max(0.0, 1.0 - (explicit_screenshots - 2) * 0.2)

            return 0.9 * first_action_score + 0.1 * excess_penalty

        return screenshot_reward

    def get_thinking_quality_reward_func(self) -> Callable:
        """
        Returns a reward function that evaluates thinking quality based on length.
    ).
        """

        def thinking_reward(completion: list[dict[str, str]], **kwargs) -> float:
            thinking_lengths = []

            for msg in completion:
                if msg["role"] == "assistant":
                    parsed = self.parser.parse(msg["content"])
                    if hasattr(parsed, "think") and parsed.think:
                        thinking_lengths.append(len(parsed.think))

            if not thinking_lengths:
                return 0.0

            avg_length = sum(thinking_lengths) / len(thinking_lengths)

            if avg_length < 20:
                return avg_length / 20
            elif avg_length <= 100:
                return 1.0
            elif avg_length <= 150:
                return 0.75
            elif avg_length <= 200:
                return 0.5
            else:
                return max(0.3, 1.0 - (avg_length - 200) / 400)

        return thinking_reward
