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
        action_mappings: dict[str, Any] | None = None,
        xml_weight: float = 0.6,
        action_weight: float = 0.4,
    ):
        """Initialize the ToolXMLParser.
        Args:
            fields: XML fields to parse
            action_mappings: Action mapping configuration from environment config
            xml_weight: Weight for XML format score (default 0.6)
            action_weight: Weight for action syntax score (default 0.4)
        """
        super().__init__(fields)
        self.action_mappings = action_mappings or {}
        self.xml_weight = xml_weight
        self.action_weight = action_weight
        # Normalize weights
        total = self.xml_weight + self.action_weight
        if total > 0:
            self.xml_weight /= total
            self.action_weight /= total

    def parse(self, text: str, strip: bool = True) -> Any:
        """Parse XML and validate action syntax if tool tag present."""
        result = super().parse(text, strip)

        # Check if "think" field exists in parsed result
        if hasattr(result, "think"):
            # Remove think blocks before parsing for tool tags
            text_no_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            tool_parse = super().parse(text_no_think, strip)
            result.tool = tool_parse.tool if hasattr(tool_parse, "tool") else None

        # If there's a tool tag, parse and store the action
        if hasattr(result, "tool") and result.tool:
            try:
                result.action = self._parse_action(result.tool)
            except ValueError as e:
                result.action = None
                result.action_error = str(e)

        return result

    def _parse_action(self, call_str: str) -> dict[str, Any]:
        """Parse function call syntax into action dict using config.

        Uses action_mappings to understand expected arguments for each tool.
        Falls back to generic parsing if tool not in mappings.
        """
        # Match function name and arguments
        match = re.match(r"(\w+)\((.*)\)", call_str.strip())
        if not match:
            raise ValueError(f"Invalid function call syntax: {call_str}")

        action_name = match.group(1)
        args_str = match.group(2).strip()

        # Parse arguments
        if not args_str:
            # No arguments (e.g., screenshot(), done())
            return {"name": action_name, "arguments": {}}

        # Parse the argument string into a list of values
        args = self._parse_argument_string(args_str)

        # Get positional argument names from config if available
        positional_names = []
        if action_name in self.action_mappings:
            parser_config = self.action_mappings[action_name].get("_parser", {})
            positional_names = parser_config.get("positional", [])

        # Map positional arguments to named arguments
        arguments = {}
        for i, arg_value in enumerate(args):
            if i < len(positional_names):
                arg_name = positional_names[i]
            else:
                # Fallback to generic naming if not in config
                arg_name = f"arg{i}"
            arguments[arg_name] = arg_value

        return {"name": action_name, "arguments": arguments}

    def _parse_argument_string(self, args_str: str) -> list[Any]:
        """Parse comma-separated arguments, handling quoted strings and numbers."""
        if not args_str:
            return []

        args = []
        current_arg = ""
        in_quotes = False
        quote_char = None
        paren_depth = 0

        for char in args_str:
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
                current_arg += char
            elif char == quote_char and in_quotes and paren_depth == 0:
                in_quotes = False
                quote_char = None
                current_arg += char
            elif char == "(" and not in_quotes:
                paren_depth += 1
                current_arg += char
            elif char == ")" and not in_quotes:
                paren_depth -= 1
                current_arg += char
            elif char == "," and not in_quotes and paren_depth == 0:
                args.append(self._parse_single_arg(current_arg.strip()))
                current_arg = ""
            else:
                current_arg += char

        # Don't forget the last argument
        if current_arg:
            args.append(self._parse_single_arg(current_arg.strip()))

        return args

    def _parse_single_arg(self, arg: str) -> Any:
        """Parse a single argument value."""
        # Remove quotes if present
        if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
            return arg[1:-1]

        # Try to parse as number
        try:
            if "." in arg:
                return float(arg)
            else:
                return int(arg)
        except ValueError:
            # Keep as string
            return arg

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
