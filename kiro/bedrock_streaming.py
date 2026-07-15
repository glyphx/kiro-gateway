"""
Bedrock response stream to Anthropic SSE adapter.

Converts Bedrock EventStream chunks (already in Anthropic JSON format)
into SSE text suitable for streaming back to Claude Code.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict

from loguru import logger


def format_sse(event_type: str, data: Dict[str, Any]) -> str:
    """Format a single SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_bedrock_to_anthropic_sse(
    chunks: AsyncGenerator[Dict[str, Any], None],
) -> AsyncGenerator[str, None]:
    """
    Convert Bedrock response chunks into Anthropic SSE text.

    Bedrock's Claude response chunks are already in Anthropic format:
    {"type": "message_start", "message": {...}}
    {"type": "content_block_start", ...}
    {"type": "content_block_delta", ...}
    etc.

    We just wrap each as an SSE event line.
    """
    try:
        async for chunk in chunks:
            event_type = chunk.get("type")
            if not event_type:
                logger.debug(f"Bedrock: chunk missing type field: {chunk}")
                continue
            yield format_sse(event_type, chunk)
    except Exception as e:
        logger.error(f"Bedrock streaming error: {e}")
        error_payload = {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        }
        yield format_sse("error", error_payload)


async def collect_bedrock_response(
    chunks: AsyncGenerator[Dict[str, Any], None],
) -> Dict[str, Any]:
    """
    Collect a full Bedrock streaming response into a single Anthropic response dict.

    Assembles content blocks from the stream events into the final message shape.
    """
    message = {}
    content_blocks: list = []
    stop_reason = None
    usage = {}

    async for chunk in chunks:
        event_type = chunk.get("type")

        if event_type == "message_start":
            message = chunk.get("message", {})
            content_blocks = []

        elif event_type == "content_block_start":
            content_blocks.append(chunk.get("content_block", {}))

        elif event_type == "content_block_delta":
            idx = chunk.get("index", 0)
            delta = chunk.get("delta", {})
            if idx < len(content_blocks):
                block = content_blocks[idx]
                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    block["text"] = block.get("text", "") + delta.get("text", "")
                elif delta_type == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
                elif delta_type == "input_json_delta":
                    block.setdefault("input", "")
                    block["input"] += delta.get("partial_json", "")

        elif event_type == "content_block_stop":
            idx = chunk.get("index", 0)
            if idx < len(content_blocks):
                block = content_blocks[idx]
                if block.get("type") == "tool_use" and isinstance(block.get("input"), str):
                    try:
                        block["input"] = json.loads(block["input"])
                    except (json.JSONDecodeError, TypeError):
                        block["input"] = {}

        elif event_type == "message_delta":
            delta = chunk.get("delta", {})
            stop_reason = delta.get("stop_reason", stop_reason)
            chunk_usage = chunk.get("usage", {})
            if chunk_usage:
                usage.update(chunk_usage)

    message["content"] = content_blocks
    if stop_reason:
        message["stop_reason"] = stop_reason
    if usage:
        message.setdefault("usage", {}).update(usage)

    return message
