"""OpenAI Responses request mapping for the shared deferred batch model.

Reasoning output items are carried through Strands as opaque signatures and
replayed in order before their following item. Request bodies are validated at
the JSONL boundary by ``OpenAIResponsesRequest``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from strands.types.content import Messages, SystemContentBlock
from strands.types.tools import ToolChoice, ToolChoiceToolDict, ToolSpec

from program_set.batch_json import canonical_json
from program_set.batch_transport import BatchTransport
from program_set.deferred_batch_model import DeferredBatchModel
from program_set.openai_responses_request import (
    OpenAIResponsesRequest,
    decode_reasoning_item,
)


class OpenAIBatchBackedModel(DeferredBatchModel):
    """Strands ``Model`` adapter backed by OpenAI Responses batches."""

    def __init__(
        self,
        transport: BatchTransport,
        *,
        agent_id: str,
        model_id: str,
        max_output_tokens: int = 4096,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be a non-empty model identifier")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        super().__init__(
            transport,
            agent_id=agent_id,
            config={"model": model_id, "max_output_tokens": max_output_tokens},
            custom_id_prefix="openai-custom",
            error_prefix="OpenAI batch request",
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
        return _responses_params(
            messages,
            tool_specs,
            system_prompt,
            system_prompt_content,
            tool_choice,
            self.config,
            request_kwargs,
        )


def _responses_params(
    messages: Messages,
    tool_specs: list[ToolSpec] | None,
    system_prompt: str | None,
    system_prompt_content: list[SystemContentBlock] | None,
    tool_choice: ToolChoice | None,
    model_config: Mapping[str, Any],
    extra_params: Mapping[str, Any],
) -> dict[str, object]:
    text_config = extra_params.get("text", model_config.get("text"))
    if (
        "response_format" in model_config
        or "response_format" in extra_params
        or (isinstance(text_config, Mapping) and "format" in text_config)
    ):
        raise NotImplementedError(
            "structured output is not supported in this experiment"
        )

    params: dict[str, object] = dict(model_config)
    params["input"] = _responses_input(messages)
    instructions = _instructions(system_prompt, system_prompt_content)
    if instructions:
        params["instructions"] = instructions

    if tool_specs:
        params["tools"] = [_responses_tool(spec) for spec in tool_specs]
    if tool_choice:
        if "auto" in tool_choice:
            params["tool_choice"] = "auto"
        elif "any" in tool_choice:
            params["tool_choice"] = "required"
        else:
            forced = cast(ToolChoiceToolDict, tool_choice)
            params["tool_choice"] = {
                "type": "function",
                "name": forced["tool"]["name"],
            }
    params.update(extra_params)
    request = OpenAIResponsesRequest.model_validate(params)
    return request.model_dump(mode="json", exclude_none=True)


def _responses_input(messages: Messages) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant", "system", "developer"}:
            raise ValueError(f"unsupported Strands message role {role!r}")
        content = message.get("content")
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise TypeError("Strands message content must be text or a block list")

        if role in {"system", "developer"}:
            converted.append({"role": role, "content": _text_blocks(content)})
            continue

        text_parts: list[str] = []
        emitted = False
        for block in content:
            if "text" in block:
                text = block["text"]
                if not isinstance(text, str):
                    raise ValueError("text content blocks must contain strings")
                text_parts.append(text)
            elif role == "assistant" and "reasoningContent" in block:
                if text_parts:
                    converted.append(
                        {"role": "assistant", "content": "".join(text_parts)}
                    )
                    text_parts = []
                reasoning_content = block["reasoningContent"]
                reasoning_text = (
                    reasoning_content.get("reasoningText")
                    if isinstance(reasoning_content, Mapping)
                    else None
                )
                signature = (
                    reasoning_text.get("signature")
                    if isinstance(reasoning_text, Mapping)
                    else None
                )
                if not isinstance(signature, str):
                    raise NotImplementedError(
                        "reasoning content without an opaque OpenAI signature"
                    )
                try:
                    reasoning = decode_reasoning_item(signature)
                except ValueError:
                    raise NotImplementedError(
                        "reasoning signature was not produced by this adapter"
                    ) from None
                converted.append(reasoning.model_dump(mode="json", exclude_none=True))
                emitted = True
            elif role == "assistant" and "toolUse" in block:
                if text_parts:
                    converted.append(
                        {"role": "assistant", "content": "".join(text_parts)}
                    )
                    text_parts = []
                tool_use = block["toolUse"]
                converted.append(
                    {
                        "type": "function_call",
                        "call_id": tool_use["toolUseId"],
                        "name": tool_use["name"],
                        "arguments": canonical_json(tool_use["input"]),
                    }
                )
                emitted = True
            elif role == "user" and "toolResult" in block:
                if text_parts:
                    converted.append({"role": "user", "content": "".join(text_parts)})
                    text_parts = []
                tool_result = block["toolResult"]
                converted.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_result["toolUseId"],
                        "output": _tool_result_text(tool_result["content"]),
                    }
                )
                emitted = True
            else:
                raise NotImplementedError(
                    f"unsupported {role} content block: {block!r}"
                )

        if text_parts or not emitted:
            converted.append({"role": role, "content": "".join(text_parts)})
    return converted


def _instructions(
    system_prompt: str | None,
    system_prompt_content: list[SystemContentBlock] | None,
) -> str | None:
    if system_prompt_content:
        return _text_blocks(system_prompt_content)
    return system_prompt


def _text_blocks(blocks: Sequence[Mapping[str, object]]) -> str:
    parts: list[str] = []
    for block in blocks:
        text = block.get("text")
        if not isinstance(text, str):
            raise NotImplementedError(f"unsupported text content block: {block!r}")
        parts.append(text)
    return "".join(parts)


def _tool_result_text(content: object) -> str:
    if not isinstance(content, list):
        raise NotImplementedError("tool result content must be a text block list")
    return _text_blocks(content)


def _responses_tool(spec: ToolSpec) -> dict[str, object]:
    schema = spec["inputSchema"]
    return {
        "type": "function",
        "name": spec["name"],
        "description": spec.get("description", ""),
        "parameters": schema.get("json", schema),
    }
