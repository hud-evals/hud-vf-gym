"""HUD Gym environment using native OpenAI tool calling."""

import json
import logging
import os

import hud
import verifiers as vf
import yaml
from datasets import Dataset
from hud.agents import GenericOpenAIChatAgent
from hud.datasets import Task
from openai import AsyncOpenAI
from verifiers import Info, Messages, SamplingArgs, State

from .rubrics import HUDBaseRubric

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
        allowed_tools = self.config.get("allowed_tools")

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

    # Helper: map verifiers SamplingArgs (chat) to OpenAI completion kwargs
    @staticmethod
    def _sampling_to_completion_kwargs(sampling_args: dict | SamplingArgs | None) -> dict:
        if not sampling_args:
            return {}
        if isinstance(sampling_args, dict):
            clean = {k: v for k, v in sampling_args.items() if v is not None}
        else:
            # Try attributes/fields; ignore Nones
            src = getattr(sampling_args, "__dict__", {})
            clean = {k: v for k, v in src.items() if v is not None}

        # Chat-specific normalization: max_tokens -> max_completion_tokens
        if "max_tokens" in clean:
            val = clean.pop("max_tokens")
            if val is not None:
                clean["max_completion_tokens"] = val
        return clean

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

        try:
            with hud.trace(f"rollout_{task}", job_id=self.job_id):
                # Create the agent and run the full lifecycle via agent.run
                completion_kwargs = self._sampling_to_completion_kwargs(sampling_args)

                self.logger.debug(f"Configuration: allowed_tools={self.config['allowed_tools']}, max_turns={self.max_turns}")

                agent = GenericOpenAIChatAgent(
                    openai_client=client,
                    model_name=model,
                    parallel_tool_calls=False,
                    system_prompt=self.system_prompt,
                    append_setup_output=False, # prompts can't be modified after initialization
                    allowed_tools=self.config["allowed_tools"],
                    completion_kwargs=completion_kwargs,
                )
                agent.metadata = {}

                self.logger.info(f"Running task: {hud_task.prompt}")
                trace = await agent.run(hud_task, max_steps=self.max_turns)

                # Store trace and reward for rubric evaluation
                state["trace"] = trace
                state["reward"] = trace.reward

                # Extract conversation from the agent
                completion = []

                state["prompt"] = agent.conversation_history[:2]

                if hasattr(agent, "conversation_history") and len(agent.conversation_history) >= 2:
                    for msg in agent.conversation_history[2:]:
                        formatted_msg = {"role": msg.get("role"), "content": msg.get("content", "") or ""}
                        if "tool_calls" in msg:
                            formatted_msg["tool_calls"] = msg["tool_calls"]
                        completion.append(formatted_msg)

                    self.logger.debug(f"Extracted {len(completion)} completion messages")

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

    def __del__(self):
        """Cleanup method to update job status when HUDGym is destroyed."""
        if hasattr(self, "job") and self.job:
            try:
                self.job.update_status_sync("completed")
            except Exception:
                # Silently fail since we're in __del__
                pass
