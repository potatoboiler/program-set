"""Anthropic Messages request mapping for the shared deferred batch model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from strands.types.content import Messages, SystemContentBlock
from strands.types.tools import ToolChoice, ToolChoiceToolDict, ToolSpec

from program_set.batch_transport import BatchTransport
from program_set.deferred_batch_model import DeferredBatchModel


class BatchBackedModel(DeferredBatchModel):
    """Strands model adapter backed by Anthropic Messages batches."""

    def __init__(
        self,
        transport: BatchTransport,
        *,
        agent_id: str,
        model_id: str,
        max_tokens: int = 4096,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be a non-empty model identifier")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        super().__init__(
            transport,
            agent_id=agent_id,
            config={"model": model_id, "max_tokens": max_tokens},
            custom_id_prefix="anthropic-custom",
            error_prefix="batch request",
        )

    def _request_params(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None,
        system_prompt: str | None,
        system_prompt_content: list[SystemContentBlock] | None,
        tool_choice: ToolChoice | None,
        request_kwargs: Mapping[str, Any],
    ) -> dict[str, object]:
        return _anthropic_params(
            messages,
            tool_specs,
            system_prompt,
            system_prompt_content,
            tool_choice,
            self.config,
            request_kwargs,
        )


def _anthropic_params(
    messages: Messages,
    tool_specs: list[ToolSpec] | None,
    system_prompt: str | None,
    system_prompt_content: list[SystemContentBlock] | None,
    tool_choice: ToolChoice | None,
    model_config: Mapping[str, Any],
    extra_params: Mapping[str, Any],
) -> dict[str, object]:
    params: dict[str, object] = dict(model_config)
    params["messages"] = [_anthropic_message(message) for message in messages]
    if system_prompt_content:
        params["system"] = [
            _anthropic_content_block(block) for block in system_prompt_content
        ]
    elif system_prompt:
        params["system"] = system_prompt
    if tool_specs:
        params["tools"] = [
            {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "input_schema": spec["inputSchema"].get("json", spec["inputSchema"]),
            }
            for spec in tool_specs
        ]
    if tool_choice:
        if "auto" in tool_choice:
            params["tool_choice"] = {"type": "auto"}
        elif "any" in tool_choice:
            params["tool_choice"] = {"type": "any"}
        else:
            forced = cast(ToolChoiceToolDict, tool_choice)
            params["tool_choice"] = {
                "type": "tool",
                "name": forced["tool"]["name"],
            }
    params.update(extra_params)
    return params


def _anthropic_message(message: Mapping[str, Any]) -> dict[str, object]:
    content = message["content"]
    if isinstance(content, str):
        return {"role": message["role"], "content": content}
    return {
        "role": message["role"],
        "content": [_anthropic_content_block(block) for block in content],
    }


def _anthropic_content_block(block: Mapping[str, Any]) -> dict[str, object]:
    if "text" in block:
        return {"type": "text", "text": block["text"]}
    if "toolUse" in block:
        tool_use = block["toolUse"]
        return {
            "type": "tool_use",
            "id": tool_use["toolUseId"],
            "name": tool_use["name"],
            "input": tool_use["input"],
        }
    if "toolResult" in block:
        tool_result = block["toolResult"]
        result: dict[str, object] = {
            "type": "tool_result",
            "tool_use_id": tool_result["toolUseId"],
            "content": [
                {"type": "text", "text": block["text"]} if "text" in block else block
                for block in tool_result["content"]
            ],
        }
        if tool_result.get("status") == "error":
            result["is_error"] = True
        return result
    raise AssertionError(f"unsupported Strands content block: {block!r}")
