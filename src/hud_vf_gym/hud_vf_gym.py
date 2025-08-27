"""HUD Gym environment using XML format for tool calls with MCP backend."""

import json
import os
from copy import deepcopy

import hud
import verifiers as vf
import yaml
from datasets import Dataset
from hud.clients import MCPClient
from hud.datasets import Task
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from verifiers import ChatMessage, Info, Messages, SamplingArgs, State
from verifiers.parsers.xml_parser import XMLParser

from .mcp_utils import execute_tool
from .parsers import ToolXMLParser
from .rubrics import HUDBaseRubric


class HUDGym(vf.MultiTurnEnv):
    """HUD environment using XML format for tool calls with MCP backend."""

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

        parser_config = self.config.get("parser", {})

        if not parser_config["use_thinking"]:
            fields = ["tool"]
        else:
            fields = ["think", "tool"]

        self.tool_parser = ToolXMLParser(
            fields=fields,
            action_mappings=self.config.get("action_mappings", {}),
            xml_weight=parser_config.get("xml_weight", 0.6),
            action_weight=parser_config.get("action_weight", 0.4),
        )
        self.result_parser = XMLParser(fields=["result"])

        rubric_config = self.config.get("rubric", {})
        rubric_weights = rubric_config.get("weights", None)

        rubric = HUDBaseRubric(parser=self.tool_parser, weights=rubric_weights)

        super().__init__(
            dataset=dataset,
            parser=self.tool_parser,
            rubric=rubric,
            system_prompt=system_prompt,
            max_turns=max_turns,
            **kwargs,
        )

    async def setup_state(self, state: State, **kwargs) -> State:
        """Setup initial state with tool tracking."""

        state = await super().setup_state(state, **kwargs)

        state["error"] = None
        state["error_step"] = None
        state["tool_attempts"] = 0
        state["tool_successes"] = 0
        state["tool_errors"] = []

        return state

    @hud.instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_model_response(self, **kwargs):
        """Override get_model_response with HUD instrumentation to capture model responses."""
        return await super().get_model_response(**kwargs)

    def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """Check if the task is completed."""
        # Check if done tool was called in the last assistant message
        if isinstance(messages, list) and messages:
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    try:
                        parsed = self.tool_parser.parse(str(msg.get("content", "")))
                        if hasattr(parsed, "action") and parsed.action:
                            if parsed.action.get("name") == "done":
                                return True
                    except (ValueError, AttributeError):
                        pass
                    break

        # Also check if we've hit max turns
        return False

    def env_response(self, messages: Messages, state: State, **kwargs) -> tuple[Messages, State]:
        """Generate environment response based on the last model action."""
        # Get the last assistant message
        assert isinstance(messages, list)
        last_message = messages[-1]
        assert last_message["role"] == "assistant"

        # Extract tool from response
        response_text = str(last_message.get("content", ""))

        # Parse for tool call
        parsed = self.tool_parser.parse(response_text)
        if not (hasattr(parsed, "tool") and parsed.tool):
            return [{"role": "user", "content": "Missing Tool Call"}], state

        # Check if action was successfully parsed
        if not hasattr(parsed, "action") or parsed.action is None:
            return [{"role": "user", "content": "Invalid Tool Call Format"}], state

        # Track tool attempt
        state["tool_attempts"] = state.get("tool_attempts", 0) + 1

        # Store the action for async execution in rollout
        state["pending_action"] = parsed.action

        # Return empty to continue - action will be executed in rollout
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
        """Generate a multi-turn rollout with MCP backend."""

        self.logger.info(f"Starting rollout for task: {task}")

        is_completed = False
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

        assert isinstance(prompt, list)
        completion: list[ChatMessage] = []
        rollout = deepcopy(prompt)

        # Extract HUD-specific data from info dict (all stored as JSON strings)
        task_info = info or {}

        # Create Task to resolve env vars in mcp_config (only it has ${ENV_VAR} templates)
        task_config = Task(
            prompt="",
            mcp_config=json.loads(task_info["mcp_config"]),
            setup_tool=json.loads(task_info["setup_tool"]) if task_info.get("setup_tool") else None,
            evaluate_tool=json.loads(task_info["evaluate_tool"]) if task_info.get("evaluate_tool") else None,
        )

        mcp_config = task_config.mcp_config
        setup_tool = task_config.setup_tool
        evaluate_tool = task_config.evaluate_tool

        mcp_client = None

        try:
            with hud.trace(f"rollout_{task}", job_id=self.job_id):
                assert mcp_config, "mcp_config must be provided"
                mcp_client = MCPClient(mcp_config=mcp_config)
                self.logger.info(f"Initializing MCP client with config: {mcp_config}")
                await mcp_client.initialize()
                self.logger.info("MCP client initialized successfully")

                assert setup_tool, "setup_tool must be provided"

                # Handle both single tool and list of tools
                setup_tools = setup_tool if isinstance(setup_tool, list) else [setup_tool]

                setup_result = None
                for tool in setup_tools:
                    self.logger.info(f"Running setup tool: {tool}")
                    setup_result = await execute_tool(tool, mcp_client)
                    if not setup_result["success"]:
                        raise RuntimeError(f"Setup tool failed: {setup_result['text']}")

                # Add setup result to the last user message in the prompt
                if setup_result and setup_result.get("text"):
                    for i in range(len(rollout) - 1, -1, -1):
                        if rollout[i].get("role") == "user":
                            rollout[i]["content"] = f"{rollout[i]['content']}\n\n{setup_result['text']}"
                            # Also update the state prompt to match
                            state["prompt"][i]["content"] = rollout[i]["content"]
                            break

                turn = 0
                while not is_completed and turn < self.max_turns:
                    state["turn"] = turn

                    # Get model response
                    response = await self.get_model_response(
                        prompt=rollout,
                        client=client,
                        model=model,
                        oai_tools=info.get("oai_tools", None) if info else None,
                        sampling_args=sampling_args or {},
                        message_type="chat",
                        images=kwargs.get("images"),
                    )
                    state["responses"].append(response)

                    assert isinstance(response, ChatCompletion)
                    response_text = response.choices[0].message.content
                    if not response_text:
                        raise ValueError("Model returned empty response")

                    response_message: ChatMessage = {"role": "assistant", "content": response_text}
                    rollout.append(response_message)
                    completion.append(response_message)

                    env_messages, state = self.env_response(rollout, state)

                    if env_messages and "pending_action" not in state:
                        assert isinstance(env_messages, list)
                        for msg in env_messages:
                            rollout.append(msg)
                            completion.append(msg)

                    elif "pending_action" in state:
                        action_dict = state.pop("pending_action")

                        tool_result = await execute_tool(
                            action_dict,
                            mcp_client,
                            self.config.get("action_mappings"),
                        )

                        result_text = tool_result["text"]
                        result_image = tool_result.get("image")

                        if tool_result["success"]:
                            state["tool_successes"] = state.get("tool_successes", 0) + 1

                            if result_image:
                                tool_result_message: ChatMessage = {
                                    "role": "user",
                                    "content": [
                                        # {"type": "text", "text": self.result_parser.format(result=result_text)}, #TODO: should this be configuarable?
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/png;base64,{result_image}"},
                                        },
                                    ],
                                }
                            else:
                                tool_result_message: ChatMessage = {
                                    "role": "user",
                                    "content": self.result_parser.format(result=result_text),
                                }
                        else:
                            tool_result_message: ChatMessage = {
                                "role": "user",
                                "content": "Invalid Tool Call",
                            }

                        rollout.append(tool_result_message)
                        completion.append(tool_result_message)

                        # Check if task is complete
                        if action_dict.get("name") == "done":
                            is_completed = True
                            break

                    turn += 1

                    if turn >= self.max_turns:
                        self.logger.warning(f"Task {task} reached max_turns ({self.max_turns}) without completion")
                        break

                assert evaluate_tool, "evaluate_tool must be provided in task info"

                # Handle both single tool and list of tools
                evaluate_tools = evaluate_tool if isinstance(evaluate_tool, list) else [evaluate_tool]

                eval_result = None
                for tool in evaluate_tools:
                    self.logger.info(f"Running evaluate tool: {tool}")
                    eval_result = await execute_tool(tool, mcp_client)
                    if not eval_result["success"]:
                        self.logger.warning(f"Evaluate tool failed: {eval_result['text']}")

                # Handle the evaluation result
                if eval_result and eval_result["success"]:
                    # Check if we have structured data with grade or reward
                    if eval_result["data"] and isinstance(eval_result["data"], dict):
                        # Check for both "grade" and "reward" fields
                        if "grade" in eval_result["data"]:
                            state["reward"] = float(eval_result["data"]["grade"])
                            self.logger.info(f"Task {task} evaluation grade: {state['reward']:.2f}")
                        elif "reward" in eval_result["data"]:
                            state["reward"] = float(eval_result["data"]["reward"])
                            self.logger.info(f"Task {task} evaluation reward: {state['reward']:.2f}")
                        else:
                            # No grade/reward available, but evaluation succeeded
                            self.logger.warning(f"Evaluation succeeded but no grade/reward found: {eval_result}")
                    else:
                        # No structured data available
                        self.logger.warning(f"Evaluation succeeded but no structured data: {eval_result}")
                else:
                    # Evaluation failed or no result
                    if eval_result:
                        self.logger.error(f"Evaluation failed: {eval_result['text']}")
                    else:
                        self.logger.error("Evaluation failed: No result returned")
                    state["reward"] = 0.0

                if is_completed:
                    self.logger.info(f"Task {task} completed in {turn} turns")
                else:
                    self.logger.info(f"Task {task} not completed after {turn} turns")

                state["completion"] = completion

                return completion, state

        except Exception as e:
            self.logger.error(f"Error during rollout: {e}")
            state["error"] = str(e)
            state["error_step"] = f"turn_{state.get('turn', 0)}"
            if "reward" not in state:
                state["reward"] = 0.0

            self.logger.warning(f"Task {task} failed on turn {state.get('turn', 0) + 1} with error: {e}")

            # Set completion for failed tasks
            state["completion"] = completion

            return completion, state

        finally:
            if mcp_client:
                try:
                    await mcp_client.shutdown()
                except Exception as e:
                    self.logger.error(f"Error during MCP cleanup: {e}")

    def __del__(self):
        """Cleanup method to update job status when HUDGym is destroyed."""
        if hasattr(self, "job") and self.job:
            try:
                self.job.update_status_sync("completed")
            except Exception:
                # Silently fail since we're in __del__
                pass
