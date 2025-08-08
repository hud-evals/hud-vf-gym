"""HUD Gym using HUD agents directly."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import hud
import verifiers as vf
from datasets import Dataset
from hud.datasets import TaskConfig
from hud.mcp.openai import OpenAIMCPAgent
from openai import AsyncOpenAI
from verifiers import Info, Messages, SamplingArgs, State

if TYPE_CHECKING:
    from hud.mcp.base import AgentResult


class HUDGym(vf.Environment):
    """HUD environment that delegates to HUD agents.
    
    This environment uses HUD's MCP agents directly, eliminating the need for:
    - Manual MCP client management
    - XML parsing for tool calls
    - Action mapping configuration
    - Custom tool execution logic
    
    Works with any OpenAI-compatible server (OpenAI, vLLM, Ollama, etc.)
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
            with hud.trace(f"rollout_{task}"):
                # Create HUD agent - works with any OpenAI-compatible client
                agent = OpenAIMCPAgent(
                    model_client=client,  # vLLM, Ollama, OpenAI, etc.
                    model=model,
                    mcp_config=info.get("mcp_config"),
                    # Pass through any additional MCP agent config
                    allowed_tools=info.get("allowed_tools"),
                    disallowed_tools=info.get("disallowed_tools"),
                )
                
                # Initialize MCP connection
                await agent.initialize()
                
                # Build TaskConfig from info
                task_config = TaskConfig(
                    prompt=prompt if isinstance(prompt, str) else prompt,  # Handle both str and list
                    setup_tool=info.get("setup_tool"),
                    evaluate_tool=info.get("evaluate_tool"),
                    metadata=info.get("metadata", {}),
                )
                
                # Run the complete task (setup → conversation → evaluate)
                result: AgentResult = await agent.run(task_config, max_steps=self.max_turns)
                
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