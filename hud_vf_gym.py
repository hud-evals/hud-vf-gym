"""HUD Gym environment using XML format for tool calls with MCP backend."""

import json
import logging
from copy import deepcopy
from pathlib import Path

import hud
import verifiers as vf
import yaml
from datasets import Dataset
from hud.client import MCPClient
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
        config_path: str | None = None,
        **kwargs,
    ):
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "default.yaml")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        max_turns = kwargs.pop("max_turns", self.config["defaults"]["max_turns"])
        system_prompt = kwargs.pop("system_prompt", self.config["system_prompt"])

        parser_config = self.config.get("parser", {})
        self.tool_parser = ToolXMLParser(
            fields=["think", "tool"],
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

        self.logger.setLevel(self.config.get("logging", {}).get("level", "INFO"))

    def setup_state(self, state: State, **kwargs) -> State:
        """Setup initial state with tool tracking."""

        state = super().setup_state(state, **kwargs)

        state["error"] = None
        state["error_step"] = None
        state["tool_attempts"] = 0
        state["tool_successes"] = 0
        state["tool_errors"] = []

        return state

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
        if not response_text:
            return [{"role": "user", "content": "Error: Empty response"}], state

        # Parse for tool call
        parsed = self.tool_parser.parse(response_text)
        if not (hasattr(parsed, "tool") and parsed.tool):
            error_msg = "No tool found in response. You must use a tool to interact. Expected format: <tool>action_name(args)</tool>"
            return [{"role": "user", "content": f"<result>Error: {error_msg}</result>"}], state

        # Check if action was successfully parsed
        if not hasattr(parsed, "action") or parsed.action is None:
            error_msg = getattr(parsed, "action_error", "Invalid action format")
            return [{"role": "user", "content": f"<result>Error: {error_msg}</result>"}], state

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
        state = self.setup_state(state, **kwargs)

        assert isinstance(prompt, list)
        completion: list[ChatMessage] = []
        rollout = deepcopy(prompt)

        # Extract HUD-specific data from info dict
        task_info = info or {}
        
        # Parse JSON strings back to dicts if needed
        mcp_config = task_info.get("mcp_config")
        if isinstance(mcp_config, str):
            mcp_config = json.loads(mcp_config)
            
        setup_tool = task_info.get("setup_tool")
        if isinstance(setup_tool, str):
            setup_tool = json.loads(setup_tool)
            
        evaluate_tool = task_info.get("evaluate_tool")
        if isinstance(evaluate_tool, str):
            evaluate_tool = json.loads(evaluate_tool)

        mcp_client = None

        try:
            with hud.trace(f"rollout_{task}"):
                assert mcp_config, "mcp_config must be provided"
                mcp_client = MCPClient(mcp_config=mcp_config)
                self.logger.info(f"Initializing MCP client with config: {mcp_config}")
                await mcp_client.initialize()
                self.logger.info("MCP client initialized successfully")

                assert setup_tool, "setup_tool must be provided"
                self.logger.info(f"Running setup tool: {setup_tool}")
                setup_result = await execute_tool(setup_tool, mcp_client)
                if not setup_result["success"]:
                    raise RuntimeError(f"Setup tool failed: {setup_result['text']}")

                # Add setup result as first user message if it has content
                if setup_result.get("text"):
                    setup_message: ChatMessage = {"role": "user", "content": setup_result["text"]}
                    rollout.append(setup_message)
                    completion.append(setup_message)
                    self.logger.debug(f"Added setup message to rollout: {setup_result['text'][:100]}...")

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

                    # Log assistant response
                    self.logger.debug(
                        f"Assistant: {response_text[:200]}..."
                        if len(response_text) > 200
                        else f"Assistant: {response_text}"
                    )

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
                            self.config.get("default_tool", "computer")
                        )

                        result_text = tool_result["text"]
                        result_image = tool_result.get("image")

                        if tool_result["success"]:
                            state["tool_successes"] = state.get("tool_successes", 0) + 1

                        if result_image:
                            tool_result_message: ChatMessage = {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": self.result_parser.format(result=result_text)},
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

                        rollout.append(tool_result_message)
                        completion.append(tool_result_message)

                        # Log tool result
                        if isinstance(tool_result_message["content"], list):
                            self.logger.debug(
                                f"Tool result: {result_text[:100]}... [+ image]"
                                if len(result_text) > 100
                                else f"Tool result: {result_text} [+ image]"
                            )
                        else:
                            # Text-only message
                            content = str(tool_result_message["content"])
                            self.logger.debug(
                                f"Tool result: {content[:100]}..." if len(content) > 100 else f"Tool result: {content}"
                            )

                        # Check if task is complete
                        if action_dict.get("name") == "done":
                            is_completed = True
                            break

                    turn += 1

                    if turn >= self.max_turns:
                        self.logger.warning(f"Task {task} reached max_turns ({self.max_turns}) without completion")
                        break

                assert evaluate_tool, "evaluate_tool must be provided in task info"
                eval_result = await execute_tool(evaluate_tool, mcp_client)

                # Handle the evaluation result
                if eval_result["success"]:
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
                    # Evaluation failed
                    self.logger.error(f"Evaluation failed: {eval_result['text']}")
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
                    await mcp_client.close()
                except Exception as e:
                    self.logger.error(f"Error during MCP cleanup: {e}")
