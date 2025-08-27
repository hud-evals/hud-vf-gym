"""Utility functions for MCP HUD Gym."""

import logging
from typing import Any

import mcp.types as types
from hud.types import MCPToolCall

logger = logging.getLogger(__name__)


def apply_transform(value: Any, transform_str: str, context: dict[str, Any] | None = None) -> Any:
    """Apply a transform defined as a lambda string.

    Args:
        value: The value to transform
        transform_str: Lambda string like "lambda x: x.split('+')"
        context: Optional context dict for transforms that need other arguments

    Returns:
        Transformed value
    """
    if not transform_str:
        return value

    # Create safe eval context with common functions
    safe_dict = {
        "int": int,
        "float": float,
        "str": str,
        "len": len,
        "abs": abs,
        "min": min,
        "max": max,
        # Add other safe functions as needed
    }

    try:
        # Evaluate the lambda with safe functions available
        # The lambda can use functions from safe_dict
        transform_func = eval(transform_str, {"__builtins__": {}, **safe_dict}, {})

        # Check if transform expects context
        if context is not None and "ctx" in transform_str:
            return transform_func(value, context)
        else:
            return transform_func(value)
    except Exception as e:
        logger.warning(f"Transform failed: {e}, returning original value")
        return value


def create_action_args(
    action_name: str, action_args: dict[str, Any], action_mappings: dict[str, Any]
) -> dict[str, Any] | None:
    """Create MCP tool arguments from agent action calls using config.

    Maps agent action names to the MCP tool's expected format using
    transforms defined as lambda strings in the config.
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
                    # Check if transform needs context
                    if value.get("use_context"):
                        arg_value = apply_transform(arg_value, value["transform"], action_args)
                    else:
                        arg_value = apply_transform(arg_value, value["transform"])

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


async def execute_tool(
    tool_call: dict[str, Any] | MCPToolCall, mcp_client: Any, action_mappings: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Execute a tool call through MCP.

    Handles both:
    - Direct MCP tool calls for setup/evaluate (MCPToolCall objects)
    - Agent action calls that need mapping to specified MCP tool (dicts)

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

    # Convert MCPToolCall to dict if needed
    if isinstance(tool_call, MCPToolCall):
        tool_call = tool_call.model_dump()

    # Handle 'done' action early
    if isinstance(tool_call, dict) and tool_call.get("name") == "done":
        logger.info("Done action called - task completed")
        return success_response("Agent is done, Evaluating...")

    try:
        # Check if client is initialized
        if hasattr(mcp_client, "is_connected") and not mcp_client.is_connected:
            logger.warning("MCP client not connected, attempting to initialize...")
            await mcp_client.initialize()

        tool_name = tool_call.get("name")
        tool_args = tool_call.get("arguments", {})

        if not tool_name:
            return error_response("Tool name is required")

        # Check if we should map this action through action_mappings
        if action_mappings and tool_name in action_mappings:
            # Transform the action using mappings
            logger.debug(f"Mapping action '{tool_name}' using action_mappings")
            action_name = tool_name
            mapping = action_mappings[action_name]

            # Get the MCP tool to use from _tool field (required)
            assert "_tool" in mapping, f"Action '{action_name}' missing '_tool' field in config"
            tool_name = mapping["_tool"]

            # Create the arguments for the MCP tool
            mcp_args = create_action_args(action_name, tool_args, action_mappings)
            if mcp_args is None:
                return error_response(f"Failed to map action '{action_name}'")

            tool_args = mcp_args
            logger.debug(f"Mapped to MCP tool '{tool_name}' with args: {tool_args}")
        else:
            # Direct MCP tool call (setup, evaluate, or unmapped tools)
            logger.debug(f"Direct MCP tool call: '{tool_name}' with args: {tool_args}")

        # Execute
        result = await mcp_client.call_tool(name=tool_name, arguments=tool_args)

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

        return success_response(text=text_content or "Success", image=image_data, data=structured_data)

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return error_response(str(e))
