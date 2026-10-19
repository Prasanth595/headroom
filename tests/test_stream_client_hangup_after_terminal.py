"""A client that hangs up after the terminal event still got a complete stream.

Codex CLI closes the connection as soon as it reads ``response.completed``,
before the upstream EOF ends the proxy's read loop. The outcome was decided
only after that loop, so every such turn recorded as a 502 failure with no
usage and no cache metrics, although the client got a complete 200 response.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from headroom.proxy.server import HeadroomProxy


def _event(name: str, payload: dict[str, Any]) -> bytes:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()


START = _event("response.created", {"type": "response.created", "response": {"id": "r1"}})
COMPLETED = _event(
    "response.completed",
    {
        "type": "response.completed",
        "response": {
            "id": "r1",
            "status": "completed",
            "usage": {
                "input_tokens": 1_000,
                "input_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 0},
                "output_tokens": 5,
            },
        },
    },
)


def _proxy(upstream_eof: asyncio.Event) -> HeadroomProxy:
    proxy = object.__new__(HeadroomProxy)
    proxy.http_client = MagicMock(spec=httpx.AsyncClient)
    proxy._config = MagicMock()
    proxy._config.ccr_inject_tool = False
    proxy._config.retry_max_attempts = 1
    proxy.config = proxy._config
    proxy._record_request_outcome = AsyncMock()

    response = MagicMock()
    response.headers = httpx.Headers({"content-type": "text/event-stream"})
    response.status_code = 200

    async def aiter_bytes():
        yield START
        yield COMPLETED
        await upstream_eof.wait()  # the upstream EOF the client never waits for

    response.aiter_bytes = aiter_bytes
    response.aclose = AsyncMock()
    proxy.http_client.build_request = MagicMock(return_value=MagicMock())
    proxy.http_client.send = AsyncMock(return_value=response)
    return proxy


async def _stream(proxy: HeadroomProxy):
    response = await proxy._stream_response(
        url="https://chatgpt.com/backend-api/codex/responses",
        headers={"authorization": "Bearer test"},
        body={"model": "gpt-5.4", "stream": True, "input": "Reply with: ok"},
        provider="openai",
        model="gpt-5.4",
        request_id="test-hangup",
        original_tokens=10,
        optimized_tokens=10,
        tokens_saved=0,
        transforms_applied=[],
        tags={},
        optimization_latency=0.0,
    )
    return response.body_iterator


@pytest.mark.parametrize("hangup", ["closed_at_terminal_yield", "cancelled_awaiting_eof"])
async def test_client_hangup_after_terminal_event_records_success_with_usage(hangup):
    proxy = _proxy(asyncio.Event())
    body = await _stream(proxy)
    seen = b""
    if hangup == "closed_at_terminal_yield":
        # The generator is suspended at the yield of the terminal chunk.
        while b"response.completed" not in seen:
            seen += await body.__anext__()
        await body.aclose()
    else:
        # The generator read past the terminal chunk and waits for EOF.
        async def read_all() -> None:
            nonlocal seen
            async for chunk in body:
                seen += chunk

        reader = asyncio.create_task(read_all())
        while b"response.completed" not in seen:
            await asyncio.sleep(0)
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader

    proxy._record_request_outcome.assert_awaited_once()
    outcome = proxy._record_request_outcome.await_args.args[0]
    assert outcome.status_code == 200
    assert outcome.provider_input_tokens == 1_000
    assert outcome.cache_read_tokens == 600
    assert outcome.output_tokens == 5


async def test_client_hangup_before_terminal_event_is_still_a_failure():
    proxy = _proxy(asyncio.Event())
    body = await _stream(proxy)
    assert b"response.created" in await body.__anext__()
    await body.aclose()

    outcome = proxy._record_request_outcome.await_args.args[0]
    assert outcome.status_code == 502
