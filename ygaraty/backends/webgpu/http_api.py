"""Bounded text-only OpenAI-compatible request and event translation."""

import asyncio
import json
import uuid

from fastapi import HTTPException
from ygaraty.backend_contracts.chat import (
    ChatRequest,
    EventKind,
    Message,
    Role,
)
from ygaraty.backend_contracts.models import HistoryMode

from .configuration import json_object


async def read_chat(request, model_id):
    body = bytearray()
    try:
        async with asyncio.timeout(30):
            async for part in request.stream():
                body.extend(part)
                if len(body) > 4 << 20:
                    raise HTTPException(413, "Request exceeds 4 MiB")
    except TimeoutError:
        raise HTTPException(408, "Body deadline exceeded") from None
    try:
        data = json_object(bytes(body))
        if set(data) - {
            "model",
            "messages",
            "stream",
            "max_tokens",
            "conversation_id",
            "temperature",
            "n",
        }:
            raise ValueError("Unsupported request field")
        if type(data.get("stream", False)) is not bool:
            raise ValueError("stream must be boolean")
        if type(data.get("n", 1)) is not int or data.get("n", 1) != 1:
            raise ValueError("Only n=1 is supported")
        temperature = data.get("temperature", 0)
        if type(temperature) not in (int, float) or temperature != 0:
            raise ValueError("Only temperature=0 is supported")
        raw = data["messages"]
        if not isinstance(raw, list) or any(
            not isinstance(m, dict) or set(m) != {"role", "content"}
            for m in raw
        ):
            raise ValueError("Expected text messages")
        messages = tuple(Message(Role(m["role"]), m["content"]) for m in raw)
        value = ChatRequest(
            uuid.uuid4().hex,
            data.get("conversation_id", uuid.uuid4().hex),
            data.get("model", model_id),
            HistoryMode.FULL,
            messages,
            data.get("max_tokens"),
        )
    except (ValueError, TypeError, KeyError):
        raise HTTPException(
            400, "Invalid or unsupported chat request"
        ) from None
    return value, data.get("stream", False)


def error_status(code):
    return {
        "access_revoked": 403,
        "stale_epoch": 409,
        "conversation_busy": 409,
        "overloaded": 429,
        "not_ready": 503,
        "execution_failed": 503,
    }.get(code, 400)


async def sse(handle, model):
    try:
        async for event in handle.events():
            value = dict(
                id=handle.request_id,
                object="chat.completion.chunk",
                model=model,
            )
            if event.kind == EventKind.ACCEPTED:
                value["choices"] = [
                    dict(
                        index=0,
                        delta={"role": "assistant"},
                        finish_reason=None,
                    )
                ]
            elif event.kind == EventKind.TEXT:
                value["choices"] = [
                    dict(
                        index=0,
                        delta={"content": event.text},
                        finish_reason=None,
                    )
                ]
            elif event.kind == EventKind.COMPLETED:
                value["choices"] = [
                    dict(
                        index=0,
                        delta={},
                        finish_reason=event.finish_reason.value,
                    )
                ]
                value["usage"] = usage(event.usage)
            else:
                value = {
                    "error": {
                        "code": event.failure.code.value
                        if event.failure
                        else "cancelled",
                        "message": event.failure.message
                        if event.failure
                        else "Request cancelled",
                    }
                }
            yield "data: " + json.dumps(value) + "\n\n"
        yield "data: [DONE]\n\n"
    finally:
        await asyncio.shield(handle.aclose())


def usage(value):
    return dict(
        prompt_tokens=value.prompt_tokens,
        completion_tokens=value.output_tokens,
        total_tokens=value.prompt_tokens + value.output_tokens,
    )


async def completion(handle, model):
    text = []
    characters = 0
    try:
        async for event in handle.events():
            if event.kind == EventKind.TEXT:
                text.append(event.text)
                characters += len(event.text)
                if characters > 4 << 20:
                    raise HTTPException(
                        413, "Response exceeds 4 MiB; use streaming"
                    )
            elif event.kind == EventKind.COMPLETED:
                return dict(
                    id=handle.request_id,
                    object="chat.completion",
                    model=model,
                    choices=[
                        dict(
                            index=0,
                            message=dict(
                                role="assistant", content="".join(text)
                            ),
                            finish_reason=event.finish_reason.value,
                        )
                    ],
                    usage=usage(event.usage),
                )
            elif event.kind in (EventKind.FAILED, EventKind.CANCELLED):
                raise HTTPException(
                    503,
                    event.failure.message
                    if event.failure
                    else "Request cancelled",
                )
    finally:
        await asyncio.shield(handle.aclose())
    raise HTTPException(503, "Incomplete response")
