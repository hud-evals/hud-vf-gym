"""Simple MCP Agent for OpenAI-compatible chat completions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import mcp.types as types
from hud.mcp.base import AgentResult, BaseMCPAgent, ModelResponse
from mcp.types import CallToolRequestParams as MCPToolCall
from openai import AsyncOpenAI

if TYPE_CHECKING:
    from hud.mcp.client import MCPClient

logger = logging.getLogger(__name__)


class SimpleMCPAgent(BaseMCPAgent):
    """
    Simple MCP agent that uses OpenAI-compatible chat completions API.
    """
    
    def __init__(
        self,
        model_client: AsyncOpenAI,
        model: str,
        mcp_client: MCPClient,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the simple MCP agent.
        
        Args:
            model_client: AsyncOpenAI client (works with any OpenAI-compatible server)
            model: Model name to use
            mcp_client: MCPClient instance for tool execution
            **kwargs: Additional arguments passed to BaseMCPAgent
        """
        super().__init__(mcp_client=mcp_client, **kwargs)
        self.client = model_client
        self.model = model
        self.model_name = f"simple-{model}"
    
    async def get_model_response(self, messages: list[dict]) -> ModelResponse:
        """
        Get model response using standard chat completions API.
        
        Args:
            messages: Chat messages
            
        Returns:
            ModelResponse with content and tool calls
        """
        try:
            # Get available tools for this conversation
            available_tools = self.get_available_tools()
            
            # Convert MCP tools to OpenAI tool format
            tools = []
            for tool in available_tools:
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                    }
                }
                if tool.inputSchema:
                    tool_def["function"]["parameters"] = tool.inputSchema
                tools.append(tool_def)
            
            # Make the API call
            if tools:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
            
            # Extract response
            result = ModelResponse(
                content=response.choices[0].message.content or "",
                tool_calls=[],
                done=False,
            )
            
            if response.choices[0].message.tool_calls:
                import json
                for tool_call in response.choices[0].message.tool_calls:
                    try:
                        args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {}
                    
                    mcp_call = MCPToolCall(
                        name=tool_call.function.name,
                        arguments=args,
                    )
                    result.tool_calls.append(mcp_call)
            
            # Mark as done if no tool calls
            if not result.tool_calls:
                result.done = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting model response: {e}")
            return ModelResponse(
                content=f"Error: {str(e)}",
                tool_calls=[],
                done=True,
            )
    
    async def create_initial_messages(
        self, prompt: str, screenshot: str | None = None
    ) -> list[dict]:
        """
        Create initial messages for the conversation.
        
        Args:
            prompt: User prompt
            screenshot: Optional base64 screenshot
            
        Returns:
            List of messages
        """
        messages = []
        
        # Add system prompt if configured
        system_prompt = self.get_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt with optional screenshot
        if screenshot:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot}"}
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        return messages
    
    async def _run_prompt(self, prompt: str, max_steps: int = 10) -> AgentResult:
        """
        Run the agent with a simple prompt.
        
        Args:
            prompt: The prompt to execute
            max_steps: Maximum number of steps
            
        Returns:
            AgentResult with messages and reward
        """
        try:
            # Capture initial screenshot if configured
            latest_screenshot = None
            if self.initial_screenshot:
                latest_screenshot = await self.capture_screenshot()
            
            # Create initial messages
            messages = await self.create_initial_messages(prompt, latest_screenshot)
            all_messages = messages.copy()
            
            step = 0
            while max_steps == -1 or step < max_steps:
                step += 1
                logger.info(f"Step {step}/{max_steps if max_steps > 0 else '∞'}")
                
                # Get model response
                response = await self.get_model_response(messages)
                
                # Add assistant message
                assistant_msg = {"role": "assistant", "content": response.content}
                if response.tool_calls:
                    # Add tool calls to message (simplified format)
                    tool_names = [tc.name for tc in response.tool_calls]
                    assistant_msg["content"] = f"{response.content}\n[Calling tools: {', '.join(tool_names)}]"
                
                messages.append(assistant_msg)
                all_messages.append(assistant_msg)
                
                # Execute tool calls
                if response.tool_calls:
                    tool_results = []
                    for tool_call in response.tool_calls:
                        logger.info(f"Executing tool: {tool_call.name}")
                        
                        # Execute the tool
                        result = await self.call_tool(tool_call)
                        tool_results.append(result)
                        
                        # Update latest screenshot if present
                        for content in result.content:
                            if isinstance(content, types.ImageContent):
                                latest_screenshot = content.data
                                break
                    
                    # Use our format_tool_results method to format the results
                    formatted_results = await self.format_tool_results(response.tool_calls, tool_results)
                    
                    # Add formatted results to messages
                    for formatted_msg in formatted_results:
                        messages.append(formatted_msg)
                        all_messages.append(formatted_msg)
                
                # Check if done
                if response.done:
                    logger.info("Agent completed task")
                    break
            
            # Get final content from last response or messages
            final_content = ""
            if 'response' in locals():
                final_content = response.content
            elif all_messages:
                # Extract from last assistant message
                for msg in reversed(all_messages):
                    if msg.get("role") == "assistant":
                        final_content = msg.get("content", "")
                        break
            
            return AgentResult(
                done=True,
                reward=0.0,  # Will be set by evaluate tool
                content=final_content,
                messages=all_messages,
            )
            
        except Exception as e:
            logger.error(f"Error during execution: {e}")
            return AgentResult(
                done=True,
                reward=0.0,
                error=str(e),
                messages=[],
            )
    
    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[types.CallToolResult]
    ) -> list[Any]:
        """
        Format tool results into messages for the model.
        
        Args:
            tool_calls: List of MCPToolCall objects that were executed
            tool_results: List of MCPToolResult objects from tool execution
            
        Returns:
            List of formatted messages for the model
        """
        messages = []
        
        for tool_call, result in zip(tool_calls, tool_results):
            # Extract text and image content from result
            text_content = ""
            image_content = None
            
            if not result.isError:
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        text_content += content.text
                    elif isinstance(content, types.ImageContent):
                        image_content = content.data
            else:
                text_content = f"Tool {tool_call.name} failed: Error in execution"
            
            # Format as message
            if image_content:
                # Multimodal response
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Result from {tool_call.name}: {text_content}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_content}"}
                        }
                    ]
                })
            else:
                # Text-only response
                messages.append({
                    "role": "user",
                    "content": f"Result from {tool_call.name}: {text_content}"
                })
        
        return messages