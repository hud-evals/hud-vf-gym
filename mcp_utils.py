"""Utility functions for MCP HUD Gym."""

import logging
from typing import Any

import mcp.types as types
from mcp.types import CallToolRequestParams as MCPToolCall

logger = logging.getLogger(__name__)


def create_computer_action_args(action_name: str, action_args: dict[str, Any], action_mappings: dict[str, Any]) -> dict[str, Any] | None:
    """Create MCP computer tool arguments from agent action calls.

    Maps agent action names (click, type, etc.) to the MCP computer tool's expected format.
    Returns None if action is unknown.
    """
    if action_name not in action_mappings:
        return None

    mapping = action_mappings[action_name]
    mcp_args = {}

    for key, value in mapping.items():
        # Skip internal fields (starting with _)
        if key.startswith("_"):
            continue

        if isinstance(value, dict):
            # Handle complex mappings
            if "static" in value:
                mcp_args[key] = value["static"]
            elif "from_arg" in value:
                arg_value = action_args.get(value["from_arg"], value.get("default", None))

                # Apply transform if specified
                if "transform" in value and arg_value is not None:
                    transform = value["transform"]
                    if transform == "split_plus":
                        # Split "ctrl+a" into ["ctrl", "a"]
                        mcp_args[key] = arg_value.split("+") if isinstance(arg_value, str) else [arg_value]
                    elif transform == "seconds_to_ms":
                        # Convert seconds to milliseconds
                        mcp_args[key] = int(arg_value * 1000)
                    elif transform == "direction_to_scroll_x":
                        # Get amount from action_args if available
                        amount = action_args.get("amount", mapping.get("_amount", {}).get("default", 3))
                        if arg_value == "right":
                            mcp_args[key] = amount
                        elif arg_value == "left":
                            mcp_args[key] = -amount
                        else:
                            mcp_args[key] = 0
                    elif transform == "direction_to_scroll_y":
                        # Get amount from action_args if available
                        amount = action_args.get("amount", mapping.get("_amount", {}).get("default", 3))
                        if arg_value == "down":
                            mcp_args[key] = amount
                        elif arg_value == "up":
                            mcp_args[key] = -amount
                        else:
                            mcp_args[key] = 0
                else:
                    mcp_args[key] = arg_value
            elif "from_args" in value:
                # Build list from multiple args
                defaults = value.get("defaults", [])
                mcp_args[key] = [
                    action_args.get(arg, defaults[i] if i < len(defaults) else None)
                    for i, arg in enumerate(value["from_args"])
                ]
        else:
            # Simple static mapping
            mcp_args[key] = value

    return mcp_args


async def execute_tool(tool_call: dict[str, Any] | MCPToolCall, mcp_client: Any, action_mappings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a tool call through MCP.

    Handles both:
    - Direct MCP tool calls (MCPToolCall objects) for setup/evaluate
    - Agent action calls (dict format) that need mapping to computer tool format

    Always returns:
    {
        "success": bool,
        "text": str,
        "image": str | None,
        "data": Any | None
    }
    """
    # Standard error response helper
    def error_response(text: str) -> dict[str, Any]:
        return {"success": False, "text": text, "image": None, "data": None}

    # Standard success response helper
    def success_response(text: str = "Success", image: str | None = None, data: Any = None) -> dict[str, Any]:
        return {"success": True, "text": text, "image": image, "data": data}

    if not mcp_client:
        logger.error("MCP client not initialized")
        return error_response("MCP client not initialized")

    # Handle 'done' action early
    if isinstance(tool_call, dict) and tool_call.get("name") == "done":
        logger.info("Done action called - task completed")
        return success_response("Task completed")

    try:
        # Get session
        sessions = mcp_client.get_all_active_sessions()
        if not sessions:
            logger.error("No active MCP sessions")
            return error_response("No active MCP sessions")

        session = next(iter(sessions.values()))
        if not session.connector.client_session:
            logger.error("MCP session not properly initialized")
            return error_response("MCP session not properly initialized")

        # Prepare tool call
        if isinstance(tool_call, MCPToolCall):
            # Direct MCP tool
            tool_name = tool_call.name
            tool_args = tool_call.arguments
        else:
            # Agent action - map to computer tool
            if not action_mappings:
                return error_response("Action mappings required")

            action_name = tool_call["name"]
            mcp_args = create_computer_action_args(
                action_name,
                tool_call.get("arguments", {}),
                action_mappings
            )
            if mcp_args is None:
                return error_response(f"Unknown action '{action_name}'")

            tool_name = "computer"
            tool_args = mcp_args

        # Execute
        result = await session.connector.client_session.call_tool(tool_name, tool_args)

        # Handle errors
        if result.isError:
            error_text = "Unknown error"
            if result.content and len(result.content) > 0:
                content = result.content[0]
                if isinstance(content, types.TextContent):
                    error_text = content.text
            return error_response(error_text)

        # Extract content
        text_content = ""
        image_data = None
        structured_data = None

        # Check for structured content (from MCP tools like evaluate)
        if result.structuredContent:
            structured_data = result.structuredContent
            # Also try to get text representation
            if isinstance(structured_data, dict):
                text_content = str(structured_data.get("grade", structured_data))
            else:
                text_content = str(structured_data)

        # Extract text and image content
        if result.content:
            for content in result.content:
                if isinstance(content, types.TextContent):
                    text_content = content.text
                elif isinstance(content, types.ImageContent):
                    image_data = content.data

        return success_response(
            text=text_content or "Success",
            image=image_data,
            data=structured_data
        )

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return error_response(str(e))
