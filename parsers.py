"""Parsers for HUD VF Gym environment."""

import re
from collections.abc import Callable
from typing import Any

from verifiers.parsers.xml_parser import XMLParser
from verifiers.types import ChatMessage


class ToolXMLParser(XMLParser):
    """XMLParser that also validates action syntax inside tool tags."""

    def __init__(
        self,
        fields: list[str | tuple[str, ...]],
        xml_weight: float = 0.6,
        action_weight: float = 0.4,
    ):
        """Initialize the ToolXMLParser.
        Args:
            fields: XML fields to parse
            xml_weight: Weight for XML format score (default 0.6)
            action_weight: Weight for action syntax score (default 0.4)
        """
        super().__init__(fields)
        self.xml_weight = xml_weight
        self.action_weight = action_weight
        # Normalize weights
        total = self.xml_weight + self.action_weight
        if total > 0:
            self.xml_weight /= total
            self.action_weight /= total

    def parse(self, text: str, strip: bool = True) -> Any:
        """Parse XML and validate action syntax if tool tag present."""
        # First do normal XML parsing
        result = super().parse(text, strip)

        # If there's a tool tag, parse and store the action
        if hasattr(result, 'tool') and result.tool:
            try:
                result.action = self._parse_action(result.tool)
            except ValueError as e:
                # Store error info but don't fail the parse
                result.action = None
                result.action_error = str(e)

        return result

    def _parse_action(self, call_str: str) -> dict[str, Any]:
        """Parse function call syntax into action dict.
        Examples:
            'screenshot()' -> {"name": "screenshot", "arguments": {}}
            'click(100, 200)' -> {"name": "click", "arguments": {"x": 100, "y": 200}}
            'type("hello world")' -> {"name": "type", "arguments": {"text": "hello world"}}
            'key("ctrl+a")' -> {"name": "key", "arguments": {"key": "ctrl+a"}}
            'scroll("down", 3)' -> {"name": "scroll", "arguments": {"direction": "down", "amount": 3}}
            'wait(2.5)' -> {"name": "wait", "arguments": {"seconds": 2.5}}
            'done()' -> {"name": "done", "arguments": {}}
        """
        # Match function name and arguments
        match = re.match(r'(\w+)\((.*)\)', call_str.strip())
        if not match:
            raise ValueError(f"Invalid function call syntax: {call_str}")

        action_name = match.group(1)
        args_str = match.group(2).strip()

        # Parse arguments based on action type
        if not args_str:
            # No arguments (e.g., screenshot(), done())
            return {"name": action_name, "arguments": {}}

        # Special handling for each action type
        if action_name == "click":
            # Parse: click(x, y)
            parts = args_str.split(',')
            if len(parts) != 2:
                raise ValueError(f"click() expects 2 arguments, got {len(parts)}")
            return {
                "name": "click",
                "arguments": {
                    "x": int(parts[0].strip()),
                    "y": int(parts[1].strip())
                }
            }

        elif action_name == "type":
            # Parse: type("text") or type('text')
            match = re.match(r'^["\'](.+)["\']$', args_str)
            if not match:
                raise ValueError("type() expects a quoted string argument")
            return {
                "name": "type",
                "arguments": {"text": match.group(1)}
            }

        elif action_name == "key":
            # Parse: key("key_name")
            match = re.match(r'^["\'](.+)["\']$', args_str)
            if not match:
                raise ValueError("key() expects a quoted string argument")
            return {
                "name": "key",
                "arguments": {"key": match.group(1)}
            }

        elif action_name == "scroll":
            # Parse: scroll("direction", amount)
            match = re.match(r'^["\'](\w+)["\'],\s*(\d+)$', args_str)
            if not match:
                raise ValueError("scroll() expects a quoted direction and numeric amount")
            return {
                "name": "scroll",
                "arguments": {
                    "direction": match.group(1),
                    "amount": int(match.group(2))
                }
            }

        elif action_name == "wait":
            # Parse: wait(seconds)
            try:
                seconds = float(args_str)
                return {
                    "name": "wait",
                    "arguments": {"seconds": seconds}
                }
            except ValueError as e:
                raise ValueError("wait() expects a numeric argument") from e

        else:
            raise ValueError(f"Unknown action: {action_name}")

    def get_format_reward_func(self) -> Callable:
        """Return a reward function that validates both XML format and action syntax."""
        # Get the base XML format reward function
        xml_reward_func = super().get_format_reward_func()

        def combined_format_reward_func(completion: list[ChatMessage], parser=None, **kwargs) -> float:
            """Check both XML format and action syntax."""
            # First get XML format score
            xml_score = xml_reward_func(completion)

            # Then check action syntax validity
            valid_actions = 0
            total_actions = 0

            assistant_messages = self.get_assistant_messages(completion)
            for msg in assistant_messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    try:
                        parsed = self.parse(content)
                        if hasattr(parsed, "tool") and parsed.tool:
                            total_actions += 1
                            try:
                                # Try to parse the action
                                self._parse_action(parsed.tool)
                                valid_actions += 1
                            except ValueError:
                                # Invalid action syntax
                                pass
                    except Exception:
                        # Not valid XML
                        pass

            if total_actions == 0:
                # No actions to validate, just return XML score
                return xml_score

            # Combine scores with configured weights
            action_score = valid_actions / total_actions
            return self.xml_weight * xml_score + self.action_weight * action_score

        return combined_format_reward_func
