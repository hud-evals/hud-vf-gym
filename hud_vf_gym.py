"""HUD Gym using HUD agents directly."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import hud
import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI
from verifiers import Info, Messages, SamplingArgs, State

from .mcp_agent import SimpleMCPAgent

if TYPE_CHECKING:
    from hud.mcp.base import AgentResult


class HUDGym(vf.Environment):
    """HUD environment that delegates to HUD agents.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        max_turns: int = 10,
        rubric: vf.Rubric | None = None,
        **kwargs,
    ):
        """Initialize HUD Gym.
        
        Args:
            dataset: Dataset with tasks containing MCP config in info dict
            max_turns: Maximum number of turns for task execution
            rubric: Optional rubric for scoring (defaults to HUD evaluation)
            **kwargs: Additional arguments passed to Environment
        """
        # Use default rubric if none provided
        if rubric is None:
            from .rubrics import HUDEvaluationRubric
            rubric = HUDEvaluationRubric()
        
        super().__init__(
            dataset=dataset,
            message_type="chat",  # HUD agents use chat format
            rubric=rubric,
            **kwargs
        )
        self.max_turns = max_turns
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
    
    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info | None = None,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> tuple[Messages, State]:
        """Run complete HUD agent lifecycle and return verifier-compatible results.
        
        Args:
            client: AsyncOpenAI client (works with any OpenAI-compatible server)
            model: Model name to use
            prompt: Initial messages/prompt
            answer: Expected answer (for scoring)
            task: Task identifier
            info: Task info containing MCP config, setup/evaluate tools
            sampling_args: Sampling parameters for model
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (completion messages, state dict)
        """
        info = info or {}
        
        # Initialize state for verifiers
        state: State = {
            "prompt": prompt,
            "completion": [],
            "answer": answer,
            "task": task,
            "info": info,
            "responses": [],  # For vLLM compatibility
            "turn": 0,
        }
        
        try:
            # Create MCP client from config
            from hud.mcp.client import MCPClient
            
            mcp_config = info.get("mcp_config")
            if not mcp_config:
                raise ValueError("mcp_config is required in task info")
            
            mcp_client = MCPClient(mcp_config=mcp_config)
            await mcp_client.initialize()
            
            # Create simple MCP agent - works with any OpenAI-compatible client
            agent = SimpleMCPAgent(
                model_client=client,  # vLLM, Ollama, OpenAI, etc.
                model=model,
                mcp_client=mcp_client,
                # Pass through any additional MCP agent config
                allowed_tools=info.get("allowed_tools"),
                disallowed_tools=info.get("disallowed_tools"),
            )
            
            # Initialize agent (discovers tools, etc.)
            await agent.initialize()
            
            # Handle setup phase if specified
            setup_tool = info.get("setup_tool")
            if setup_tool:
                from mcp.types import CallToolRequestParams as MCPToolCall
                
                self.logger.info(f"Running setup tool for task {task}")
                if isinstance(setup_tool, list):
                    for tool_dict in setup_tool:
                        tool_call = MCPToolCall(
                            name=tool_dict.get("name"),
                            arguments=tool_dict.get("arguments", {})
                        )
                        await agent.call_tool(tool_call)
                else:
                    tool_call = MCPToolCall(
                        name=setup_tool.get("name"),
                        arguments=setup_tool.get("arguments", {})
                    )
                    await agent.call_tool(tool_call)
            
            # Convert prompt to string for agent
            if isinstance(prompt, list):
                # Extract user messages and join them
                prompt_str = "\n".join(
                    msg.get("content", "") for msg in prompt 
                    if msg.get("role") == "user"
                )
                if not prompt_str:
                    # Fallback: use all messages
                    prompt_str = str(prompt)
            else:
                prompt_str = prompt
            
            # Run the main conversation
            result: AgentResult = await agent.run(prompt_str, max_steps=self.max_turns)
            
            # Handle evaluate phase if specified
            evaluate_tool = info.get("evaluate_tool")
            if evaluate_tool:
                from mcp.types import CallToolRequestParams as MCPToolCall
                
                self.logger.info(f"Running evaluate tool for task {task}")
                if isinstance(evaluate_tool, list):
                    for tool_dict in evaluate_tool:
                        tool_call = MCPToolCall(
                            name=tool_dict.get("name"),
                            arguments=tool_dict.get("arguments", {})
                        )
                        eval_result = await agent.call_tool(tool_call)
                else:
                    tool_call = MCPToolCall(
                        name=evaluate_tool.get("name"),
                        arguments=evaluate_tool.get("arguments", {})
                    )
                    eval_result = await agent.call_tool(tool_call)
                
                # Extract grade from evaluation if available
                if hasattr(eval_result, "structuredContent") and eval_result.structuredContent:
                    # Try to extract grade from structured content
                    try:
                        if isinstance(eval_result.structuredContent, dict):
                            result.reward = float(eval_result.structuredContent.get("grade", 0.0))
                    except (ValueError, TypeError):
                        pass
            
            # Extract completion from result
            # HUD agents return the full conversation including environment responses
            if result.messages:
                # Skip original prompt messages
                prompt_len = len(prompt) if isinstance(prompt, list) else 0
                completion = result.messages[prompt_len:]
            else:
                # Fallback if no messages in result
                completion = []
                if result.content:
                    completion.append({"role": "assistant", "content": result.content})
            
            # Update state with HUD results
            state["completion"] = completion
            state["reward"] = result.reward  # From HUD evaluation
            state["info"].update(result.info)  # Merge additional info from agent
            
            # Add metrics for rubrics
            state["done"] = result.done
            state["error"] = result.error
            state["turns_taken"] = len([m for m in completion if m.get("role") == "assistant"])
            state["max_turns"] = self.max_turns
            
            # Extract responses for vLLM compatibility
            state["responses"] = [
                msg for msg in completion 
                if msg.get("role") == "assistant"
            ]
            
            self.logger.info(
                f"Task {task} completed: reward={result.reward:.2f}, "
                f"turns={state['turns_taken']}/{self.max_turns}"
            )
            
            return completion, state
                
        except Exception as e:
            self.logger.error(f"Rollout failed for task {task}: {e}")
            state["error"] = str(e)
            state["reward"] = 0.0
            state["done"] = False
            return [], state
        
        finally:
            # Clean up MCP client if it was created
            if 'mcp_client' in locals():
                try:
                    await mcp_client.close()
                except Exception as e:
                    self.logger.warning(f"Error closing MCP client: {e}")