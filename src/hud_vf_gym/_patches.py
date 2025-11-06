"""Local adjustments to upstream verifiers helpers."""

from __future__ import annotations

import logging
from typing import Any, cast

import hud
from hud.agents.openai_chat_generic import GenericOpenAIChatAgent
from hud.types import AgentResponse
from verifiers.types import ChatCompletion, ChatMessage, State
from verifiers.utils import processing_utils as _processing_utils

try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
except Exception:  # pragma: no cover - transformers is provided in runtime env
    PreTrainedTokenizerBase = Any  # type: ignore[assignment]

_LOGGER = logging.getLogger("hud_vf_gym.verifiers")


def _relaxed_process_chat_format_vllm(
    prompt: list[ChatMessage],
    completion: list[ChatMessage],
    state: State,
    processing_class: PreTrainedTokenizerBase,
    mask_env_responses: bool = False,
) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
    """Wrapper around verifiers process_chat_format_vllm with relaxed prefix checking.

    Qwen chat templates may rewrite consecutive tool turns, so we fall back to a
    longest-common-prefix diff instead of asserting strict prefix equality.
    """

    responses: list[ChatCompletion] = state["responses"]
    responses_idx = 0
    zipped: list[tuple[ChatMessage, ChatCompletion | None]] = []
    for turn in completion:
        if turn["role"] == "assistant" and responses_idx < len(responses):
            zipped.append((turn, responses[responses_idx]))
            responses_idx += 1
        else:
            zipped.append((turn, None))
    if responses_idx != len(responses):
        _LOGGER.debug(
            "Assistant messages (%d) did not fully consume responses (%d); treating extras as env output.",
            sum(1 for msg, _ in zipped if msg.get("role") == "assistant"),
            len(responses),
        )
    assert len(zipped) == len(completion), "Length mismatch"

    oai_tools = state.get("info", {}).get("oai_tools", []) or []
    prompt_ids: list[int] = processing_class.apply_chat_template(
        conversation=prompt,  # type: ignore[arg-type]
        add_generation_prompt=True,
        tools=oai_tools,
    )
    messages_consumed: list[ChatMessage | dict] = [m for m in prompt]
    prompt_mask: list[int] = [0] * len(prompt_ids)
    completion_ids: list[int] = []
    completion_mask: list[int] = []
    completion_logprobs: list[float] = []
    i = 0

    while i < len(zipped):
        message, response = zipped[i]

        if message["role"] == "assistant" and response is not None:
            completion_turn_ids = _processing_utils.parse_chat_completion_tokens(response)
            completion_turn_mask = [1] * len(completion_turn_ids)
            completion_turn_logprobs = _processing_utils.parse_chat_completion_logprobs(response)
            completion_ids.extend(completion_turn_ids)
            completion_mask.extend(completion_turn_mask)
            completion_logprobs.extend(completion_turn_logprobs)
            messages_consumed.append(message)
            i += 1
            continue
        assert message["role"] in {"user", "tool", "assistant"}
        consecutive_messages = [message]
        j = i + 1
        while j < len(zipped) and zipped[j][0]["role"] != "assistant":
            consecutive_messages.append(zipped[j][0])
            j += 1

        token_prefix: list[int] = processing_class.apply_chat_template(
            conversation=messages_consumed,  # type: ignore[arg-type]
            add_generation_prompt=False,
            tools=oai_tools,
        )
        token_prefix_with_turn: list[int] = processing_class.apply_chat_template(
            conversation=messages_consumed + consecutive_messages,  # type: ignore[arg-type]
            add_generation_prompt=True,
            tools=oai_tools,
        )

        lcp = 0
        max_lcp = min(len(token_prefix), len(token_prefix_with_turn))
        while lcp < max_lcp and token_prefix[lcp] == token_prefix_with_turn[lcp]:
            lcp += 1

        if lcp != len(token_prefix):
            _LOGGER.debug(
                "Relaxed prefix check for %s message(s); replaced %d prior tokens, appended %d new tokens.",
                message["role"],
                len(token_prefix) - lcp,
                len(token_prefix_with_turn) - lcp,
            )

        completion_turn_ids = token_prefix_with_turn[lcp:]
        if mask_env_responses:
            completion_turn_mask = [0] * len(completion_turn_ids)
        else:
            completion_turn_mask = [1] * len(completion_turn_ids)
        completion_turn_logprobs = [0.0] * len(completion_turn_ids)

        completion_ids.extend(completion_turn_ids)
        completion_mask.extend(completion_turn_mask)
        completion_logprobs.extend(completion_turn_logprobs)
        messages_consumed.extend(consecutive_messages)
        i = j

    return (
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
        completion_logprobs,
    )


_processing_utils.process_chat_format_vllm = _relaxed_process_chat_format_vllm

# Some modules import the symbol directly (e.g., `from ... import process_chat_format_vllm`)
# so rewrite those references too.
try:
    from verifiers.envs import environment as _vf_environment

    _vf_environment.process_chat_format_vllm = _relaxed_process_chat_format_vllm  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - best-effort monkey patching
    _LOGGER.debug("Failed patching verifiers.envs.environment: %s", exc)

# Ensure our relaxed implementation is visible wherever verifiers attach the helper.
_processing_utils.process_chat_format_vllm = _relaxed_process_chat_format_vllm


@hud.instrument(
    span_type="agent",
    record_args=True,
    record_result=True,
)
async def _patched_get_response(
    self: GenericOpenAIChatAgent,
    messages: list[Any],
) -> AgentResponse:
    """Patched version that always records messages in HUD telemetry."""

    from openai.types.chat import ChatCompletionToolParam  # Imported lazily for startup perf

    tools = cast("list[ChatCompletionToolParam]", self.get_tool_schemas())
    protected_keys = {"model", "messages", "tools"}
    extra = {k: v for k, v in (self.completion_kwargs or {}).items() if k not in protected_keys}

    try:
        response = await self._invoke_chat_completion(
            messages=messages,
            tools=tools,  # type: ignore[arg-type]
            extra=extra,
        )
    except Exception as exc:  # pragma: no cover - relies on upstream transport errors
        error_content = f"Error getting response {exc}"
        if "Invalid JSON" in str(exc):
            error_content = "Invalid JSON, response was truncated"
        self.hud_console.warning_log(error_content)
        return AgentResponse(
            content=error_content,
            tool_calls=[],
            done=True,
            isError=True,
            raw=None,
        )

    choice = response.choices[0]
    msg = choice.message
    assistant_msg: dict[str, Any] = {"role": "assistant"}

    if msg.content:
        assistant_msg["content"] = msg.content

    if msg.tool_calls:
        serialized_tool_calls = []
        for tc in msg.tool_calls:
            serialized_tool_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )
        assistant_msg["tool_calls"] = serialized_tool_calls

    messages.append(assistant_msg)

    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name is not None:
                tool_calls.append(self._oai_to_mcp(tc))

    done = choice.finish_reason == "length"
    if done:
        self.hud_console.info_log(f"Done decision: finish_reason={choice.finish_reason}")

    return AgentResponse(
        content=msg.content or "",
        tool_calls=tool_calls,
        done=done,
        raw=response,
    )


GenericOpenAIChatAgent.get_response = _patched_get_response
