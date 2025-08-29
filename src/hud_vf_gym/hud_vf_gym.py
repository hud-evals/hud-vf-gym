"""HUD Gym environment using native OpenAI tool calling."""

import json
import os

import hud
import verifiers as vf
import yaml
from datasets import Dataset
from hud.agents import GenericOpenAIChatAgent
from hud.clients import MCPClient
from hud.datasets import Task
from openai import AsyncOpenAI
from verifiers import Info, Messages, SamplingArgs, State

from .rubrics import HUDBaseRubric

import logging; logging.getLogger("verifiers").setLevel(logging.DEBUG)

class HUDGym(vf.MultiTurnEnv):
    """HUD environment using native OpenAI tool calling."""

    def __init__(
        self,
        dataset: Dataset,
        config_path: str,
        **kwargs,
    ):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        max_turns = kwargs.pop("max_turns", self.config["defaults"]["max_turns"])
        system_prompt = kwargs.pop("system_prompt", self.config["system_prompt"])

        # Handle job creation from config
        job_config = self.config.get("job", {})

        # Check if HUD_API_KEY is provided
        assert os.getenv("HUD_API_KEY"), "HUD_API_KEY environment variable must be set"

        # Create the job from config
        self.job = hud.create_job(
            name=job_config.get("name", "HUDGym Run"),
            metadata=job_config.get("metadata", {}),
            dataset_link=job_config.get("dataset_link"),
        )
        self.job.update_status_sync("running")
        self.job_id = self.job.id

        # Create rubric for scoring
        rubric = HUDBaseRubric()

        super().__init__(
            dataset=dataset,
            parser=None,
            rubric=rubric,
            system_prompt=system_prompt,
            max_turns=max_turns,
            **kwargs,
        )

    async def setup_state(self, state: State, **kwargs) -> State:
        """Setup initial state."""
        state = await super().setup_state(state, **kwargs)
        
        state["error"] = None
        state["error_step"] = None
        state["trace"] = None
        
        return state

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """Check if the task is completed."""
        # With the agent approach, we rely on the trace to determine completion
        if state.get("trace"):
            return state["trace"].done
        return False

    async def env_response(self, messages: Messages, state: State, **kwargs) -> tuple[Messages, State]:
        """Generate environment response."""
        return [], state

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
        """Generate a rollout using GenericOpenAIChatAgent."""
        
        self.logger.info(f"Starting rollout for task: {task}")
        
        state: State = {
            "prompt": prompt,
            "completion": [],
            "answer": answer,
            "task": task,
            "info": info or {},
            "responses": [],
            "turn": 0,
        }
        state = await self.setup_state(state, **kwargs)
        
        # Extract HUD-specific data from info dict
        task_info = info or {}
        
        # Create Task object
        prompt_text = ""
        if prompt and isinstance(prompt, list):
            for msg in reversed(prompt):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        prompt_text = content
                    break
        
        hud_task = Task(
            prompt=prompt_text,
            mcp_config=json.loads(task_info["mcp_config"]),
            setup_tool=json.loads(task_info["setup_tool"]) if task_info.get("setup_tool") else None,
            evaluate_tool=json.loads(task_info["evaluate_tool"]) if task_info.get("evaluate_tool") else None,
            system_prompt=self.system_prompt,
        )
        
        mcp_client = None

        try:
            with hud.trace(f"rollout_{task}", job_id=self.job_id):
                # Create MCP client
                mcp_client = MCPClient(mcp_config=hud_task.mcp_config)
                
                # Create the agent
                agent = GenericOpenAIChatAgent(
                    mcp_client=mcp_client,
                    openai_client=client,
                    model_name=model,
                    parallel_tool_calls=False,
                    system_prompt=self.system_prompt,
                    append_setup_output=True,
                )
                agent.metadata = {}
                
                # MANUAL LIFECYCLE IMPLEMENTATION
                # Phase 1: Initialize agent with task context
                self.logger.info("Initializing agent...")
                await agent.initialize(hud_task)
                
                # Phase 2: Run setup tool if present
                setup_content = []
                if hud_task.setup_tool:
                    self.logger.info(f"Running setup tool: {hud_task.setup_tool}")
                    setup_results = await agent.call_tools(hud_task.setup_tool)
                    
                    # Check for errors
                    if any(result.isError for result in setup_results):
                        raise RuntimeError(f"Setup failed: {setup_results}")
                    
                    # Extract setup content if append_setup_output is True
                    if agent.append_setup_output and setup_results:
                        if isinstance(setup_results[0].content, list):
                            setup_content.extend(setup_results[0].content)
                        elif setup_results[0].content:
                            setup_content.append(setup_results[0].content)
                    
                    self.logger.info("Setup complete")
                
                # Phase 3: Build initial context and messages
                self.logger.info(f"Running task: {hud_task.prompt}")
                
                # Get system messages from agent
                messages = await agent.get_system_messages()
                
                # Build context with setup output and prompt
                from hud.agents.base import text_to_blocks
                context_blocks = []
                
                # Add setup content
                if setup_content:
                    context_blocks.extend(setup_content)
                
                # Add task prompt
                if hud_task.prompt:
                    context_blocks.extend(text_to_blocks(hud_task.prompt))
                
                # Format context into messages
                context_messages = await agent.format_message(context_blocks)
                messages.extend(context_messages)
                
                # Store conversation history
                agent.conversation_history = messages.copy()
                
                # Phase 4: Run agent execution loop
                done = False
                steps = 0
                trace_content = []
                
                while not done and steps < self.max_turns:
                    self.logger.info(f"Step {steps + 1}/{self.max_turns}")
                    
                    # Get model response - this updates messages and conversation_history
                    response = await agent.get_response(messages)
                    
                    if response.content:
                        self.logger.debug(f"Agent response: {response.content[:200]}...")
                        trace_content.append(response.content)
                    
                    if response.tool_calls:
                        # Execute tool calls
                        self.logger.debug(f"Executing {len(response.tool_calls)} tool calls")
                        tool_results = await agent.call_tools(response.tool_calls)
                        
                        # Format tool results back into messages
                        # This also updates conversation_history internally
                        tool_messages = await agent.format_tool_results(response.tool_calls, tool_results)
                        messages.extend(tool_messages)
                        
                        # Check for errors in tool results
                        if any(result.isError for result in tool_results):
                            self.logger.warning(f"Tool error at step {steps}: {tool_results}")
                    else:
                        # No more tool calls, agent is done
                        done = True
                        self.logger.info("Agent finished (no more tool calls)")
                    
                    steps += 1
                
                # Phase 5: Run evaluation if present
                eval_reward = 0.0
                eval_content = None
                if hud_task.evaluate_tool:
                    self.logger.info(f"Running evaluation tool: {hud_task.evaluate_tool}")
                    eval_results = await agent.call_tools(hud_task.evaluate_tool)
                    
                    if any(result.isError for result in eval_results):
                        self.logger.error(f"Evaluation failed: {eval_results}")
                    else:
                        # Extract reward from evaluation
                        from hud.agents.base import find_reward, find_content
                        eval_reward = find_reward(eval_results[0])
                        eval_content = find_content(eval_results[0])
                        self.logger.info(f"Evaluation complete - Reward: {eval_reward}")
                
                # Create trace object to match expected format
                from hud.agents.base import Trace
                trace = Trace(
                    reward=eval_reward,
                    done=done,
                    content=eval_content or "\n".join(trace_content),
                    isError=False
                )
                
                # Store trace in state for rubric evaluation
                state["trace"] = trace
                state["reward"] = trace.reward
                
                # Extract conversation from the agent
                completion = []
                full_conversation = []
                
                if hasattr(agent, 'conversation_history'):
                    # Process all messages from the conversation
                    for msg in agent.conversation_history:
                        if msg.get("role") == "system":
                            # System messages go to the beginning
                            full_conversation.insert(0, msg)
                        elif msg.get("role") == "tool":
                            # Convert tool messages to user messages for verifiers
                            formatted_msg = {
                                "role": "user",
                                "content": msg.get("content", "") or ""
                            }
                            full_conversation.append(formatted_msg)
                            completion.append(formatted_msg)
                        else:
                            # Assistant and user messages
                            formatted_msg = {
                                "role": msg.get("role"),
                                "content": msg.get("content", "") or ""
                            }
                            # Include tool_calls if present (keep OpenAI objects)
                            if "tool_calls" in msg:
                                formatted_msg["tool_calls"] = msg["tool_calls"]
                            full_conversation.append(formatted_msg)
                            # Only non-system messages go to completion
                            if msg.get("role") != "system":
                                completion.append(formatted_msg)
                    
                    self.logger.debug(f"Extracted {len(completion)} completion messages")
                    
                    # Update state["prompt"] to include system message
                    if full_conversation and full_conversation[0].get("role") == "system":
                        # Replace the prompt with the full conversation including system
                        state["prompt"] = full_conversation[:2]  # System + first user message
                        self.logger.debug(f"Updated state['prompt']: {state['prompt']}")

                state["completion"] = completion
                
                self.logger.info(f"Task {task} completed with reward: {trace.reward}")
                
                return completion, state
                
        except Exception as e:
            self.logger.error(f"Error during rollout: {e}")
            state["error"] = str(e)
            state["error_step"] = f"turn_{state.get('turn', 0)}"
            state["reward"] = 0.0
            
            self.logger.warning(f"Task {task} failed with error: {e}")
            
            return [], state
        
        finally:
            if mcp_client:
                try:
                    await mcp_client.shutdown()
                    self.logger.debug("MCP client shut down successfully")
                except Exception as e:
                    self.logger.warning(f"Error shutting down MCP client: {e}")


    def __del__(self):
        """Cleanup method to update job status when HUDGym is destroyed."""
        if hasattr(self, "job") and self.job:
            try:
                self.job.update_status_sync("completed")
            except Exception:
                # Silently fail since we're in __del__
                pass