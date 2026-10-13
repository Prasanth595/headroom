from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import StreamingResponse

from headroom import paths
from headroom.proxy.server import HeadroomProxy, ProxyConfig

ROOT = Path(__file__).resolve().parents[1]
_ISOLATED_MODULE_NAMES = (
    "headroom.proxy",
    "headroom.proxy.handlers",
    "fastapi.responses",
    "tests.headroom_proxy_handlers_openai",
    "tests.headroom_proxy_handlers_streaming",
)


@pytest.fixture(autouse=True)
def restore_isolated_modules() -> None:
    saved_modules = {name: sys.modules.get(name) for name in _ISOLATED_MODULE_NAMES}
    try:
        yield
    finally:
        for name in _ISOLATED_MODULE_NAMES:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module


def _load_handler_module(monkeypatch: pytest.MonkeyPatch, module_name: str, relative_path: str):
    proxy_pkg = types.ModuleType("headroom.proxy")
    proxy_pkg.__path__ = [str(ROOT / "headroom" / "proxy")]
    monkeypatch.setitem(sys.modules, "headroom.proxy", proxy_pkg)

    handlers_pkg = types.ModuleType("headroom.proxy.handlers")
    handlers_pkg.__path__ = [str(ROOT / "headroom" / "proxy" / "handlers")]
    monkeypatch.setitem(sys.modules, "headroom.proxy.handlers", handlers_pkg)

    # The real httpx, not a stub: it is a core dependency, and a partial stub
    # breaks whatever else imports httpx during a cold import (the CCR/MCP SSE
    # client needs httpx.Response). It also keeps the exception hierarchy the
    # proxy catches (TransportError and its subclasses) exactly as in production.

    responses_mod = types.ModuleType("fastapi.responses")

    class Response:
        def __init__(
            self,
            content=None,
            status_code: int = 200,
            headers=None,
            media_type=None,
            background=None,
        ):
            self.content = content
            self.status_code = status_code
            self.headers = headers or {}
            self.media_type = media_type
            # The streaming forwarder attaches a background task that releases the
            # upstream stream when the body is never consumed (#2882); the double
            # must accept and store it so the real StreamingResponse call works.
            self.background = background

    class StreamingResponse(Response):
        pass

    class JSONResponse(Response):
        pass

    responses_mod.Response = Response
    responses_mod.StreamingResponse = StreamingResponse
    responses_mod.JSONResponse = JSONResponse
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses_mod)

    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_openai_passthrough_applies_copilot_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    openai_mod = _load_handler_module(
        monkeypatch,
        "tests.headroom_proxy_handlers_openai",
        "headroom/proxy/handlers/openai.py",
    )

    seen: dict[str, object] = {}

    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        seen["headers"] = dict(headers)
        seen["url"] = url
        return {"Authorization": "Bearer upstream-token"}

    monkeypatch.setattr(openai_mod, "apply_copilot_api_auth", fake_apply)

    class Dummy(openai_mod.OpenAIHandlerMixin):
        def __init__(self) -> None:
            self.metrics = SimpleNamespace(record_request=self._record_request)
            self.http_client = SimpleNamespace(request=self._request)
            self.cost_tracker = None
            self._counter = 0

        async def _record_request(self, **kwargs) -> None:  # noqa: ANN003
            return None

        async def _next_request_id(self) -> str:
            # The passthrough handler now allocates a request_id at end-
            # of-call because it records via ``_record_request_outcome``,
            # which requires one. Pre-refactor the dummy didn't need
            # this method because metrics.record_request was called
            # directly without a request_id.
            self._counter += 1
            return f"req-{self._counter}"

        async def _record_request_outcome(self, outcome) -> None:  # noqa: ANN001
            from headroom.proxy.outcome import emit_request_outcome

            await emit_request_outcome(self, outcome)

        def _extract_tags(self, headers: dict) -> dict[str, str]:
            # Mirror of HeadroomProxy._extract_tags. The passthrough
            # handler now extracts tags at entry as part of the
            # outcome-tag invariant lock (PR #480).
            return {
                k.lower().replace("x-headroom-", ""): v
                for k, v in headers.items()
                if k.lower().startswith("x-headroom-")
            }

        async def _request(self, **kwargs):  # noqa: ANN003
            seen["request_kwargs"] = kwargs
            return SimpleNamespace(headers={}, content=b"{}", status_code=200)

    request = SimpleNamespace(
        url=SimpleNamespace(path="/v1/models", query=""),
        headers={
            "authorization": "Bearer downstream",
            "host": "localhost",
            "accept-encoding": "gzip",
        },
        method="GET",
        body=lambda: None,
    )

    async def body() -> bytes:
        return b""

    request.body = body

    handler = Dummy()
    response = asyncio.run(
        handler.handle_passthrough(
            request,
            "https://api.githubcopilot.com",
            "models",
            "openai",
        )
    )

    assert seen["url"] == "https://api.githubcopilot.com/models"
    assert seen["request_kwargs"]["headers"] == {"Authorization": "Bearer upstream-token"}
    assert response.status_code == 200


def test_streaming_response_applies_copilot_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    streaming_mod = _load_handler_module(
        monkeypatch,
        "tests.headroom_proxy_handlers_streaming",
        "headroom/proxy/handlers/streaming.py",
    )

    seen: dict[str, object] = {}

    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        seen["headers"] = dict(headers)
        seen["url"] = url
        return {"Authorization": "Bearer upstream-token"}

    monkeypatch.setattr(streaming_mod, "apply_copilot_api_auth", fake_apply)

    class Dummy(streaming_mod.StreamingMixin):
        def __init__(self) -> None:
            self.memory_handler = None
            self.config = SimpleNamespace(
                retry_max_attempts=1,
                retry_base_delay_ms=1,
                retry_max_delay_ms=1,
            )
            self.http_client = SimpleNamespace(
                build_request=self._build_request,
                send=self._send,
            )

        def _build_request(self, method: str, url: str, **kwargs):  # noqa: ANN003
            # PR-A3: streaming forwarder is byte-faithful; it now passes
            # ``content=<bytes>`` instead of ``json=<dict>``.
            seen["request"] = {
                "method": method,
                "url": url,
                **kwargs,
            }
            return SimpleNamespace()

        async def _send(self, request, stream: bool):  # noqa: ANN001, ANN003
            return SimpleNamespace(headers={}, status_code=200)

    handler = Dummy()
    response = asyncio.run(
        handler._stream_response(
            url="https://api.githubcopilot.com/v1/responses",
            headers={"authorization": "Bearer downstream"},
            body={"model": "gpt-4o"},
            provider="openai",
            model="gpt-4o",
            request_id="req-test",
            original_tokens=0,
            optimized_tokens=0,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )
    )

    assert seen["url"] == "https://api.githubcopilot.com/v1/responses"
    # PR-A3: byte-faithful forwarder always sets ``content-type`` explicitly.
    sent_headers = seen["request"]["headers"]
    assert sent_headers["Authorization"] == "Bearer upstream-token"
    assert sent_headers["content-type"] == "application/json"
    assert response.status_code == 200
    # The Copilot auth hook and the #2882 upstream-stream cleanup coexist: the
    # streaming response still carries its background release task.
    assert response.background is not None


def test_openai_chat_routes_copilot_requests_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    openai_mod = _load_handler_module(
        monkeypatch,
        "tests.headroom_proxy_handlers_openai",
        "headroom/proxy/handlers/openai.py",
    )

    copilot_base = "https://api.githubcopilot.com"
    gpt54_mini_url = openai_mod.build_copilot_upstream_url(
        copilot_base,
        openai_mod._resolve_openai_handler_path(
            {},
            handler_path=openai_mod._resolve_openai_chat_handler_path(copilot_base, "gpt-5.4-mini"),
        ),
    )
    claude_url = openai_mod.build_copilot_upstream_url(
        copilot_base,
        openai_mod._resolve_openai_handler_path(
            {},
            handler_path=openai_mod._resolve_openai_chat_handler_path(
                copilot_base, "claude-sonnet-5"
            ),
        ),
    )
    openai_url = openai_mod.build_copilot_upstream_url(
        "https://api.openai.com",
        openai_mod._resolve_openai_handler_path(
            {},
            handler_path=openai_mod._resolve_openai_chat_handler_path(
                "https://api.openai.com", "gpt-5.4-mini"
            ),
        ),
    )

    assert gpt54_mini_url == "https://api.githubcopilot.com/responses"
    assert claude_url == "https://api.githubcopilot.com/chat/completions"
    assert openai_url == "https://api.openai.com/v1/chat/completions"


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["passthrough", "streaming"])
@pytest.mark.parametrize(
    ("path", "expected_auth"),
    [
        (
            "/v1/engines/gpt-41-copilot/completions",
            "Bearer ghu_editor_seat_fixture",
        ),
        ("/chat/completions", "Bearer tid_operator_seat_fixture"),
    ],
    ids=["native-editor-seat", "chat-operator-seat"],
)
async def test_real_forwarders_enforce_credential_and_secret_boundaries(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str, path: str, expected_auth: str
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "tid_operator_seat_fixture")
    monkeypatch.setenv("GITHUB_COPILOT_PROXY_URL", "https://gateway.example.invalid/v1")
    monkeypatch.delenv("GITHUB_COPILOT_REFRESH_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(paths, "_PROCESS_STATELESS", paths._PROCESS_STATELESS)
    for name in (
        "ANTHROPIC_API_URL",
        "OPENAI_API_URL",
        "GEMINI_API_URL",
        "CLOUDCODE_API_URL",
        "VERTEX_API_URL",
    ):
        monkeypatch.setattr(HeadroomProxy, name, getattr(HeadroomProxy, name))
    proxy = HeadroomProxy(
        ProxyConfig(
            stateless=True,
            optimize=False,
            disable_kompress=True,
            discover_pipeline_extensions=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    content = b'data: {"choices":[{"text":"accepted"}]}\n\ndata: [DONE]\n\n'

    def credential_gate(request: httpx.Request) -> httpx.Response:
        admitted = (
            request.headers.get("authorization") == expected_auth
            and request.headers.get("copilot-integration-id") == "vscode"
            and "x-api-key" not in request.headers
        )
        return httpx.Response(
            200 if admitted else 401,
            content=content if admitted else b'{"error":"wrong credential or leaked secret"}',
            headers={"content-type": "text/event-stream" if admitted else "application/json"},
        )

    headers = {
        "authorization": "Bearer ghu_editor_seat_fixture",
        "copilot-integration-id": "vscode",
        "x-api-key": "unrelated-provider-secret",
    }
    base = (
        "https://proxy.business.githubcopilot.com"
        if path.startswith("/v1/engines/")
        else "https://api.githubcopilot.com"
    )
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(credential_gate)) as client:
            proxy.http_client = client
            if entrypoint == "passthrough":

                async def receive() -> dict[str, object]:
                    return {"type": "http.request", "body": b"{}", "more_body": False}

                request = Request(
                    {
                        "type": "http",
                        "method": "POST",
                        "scheme": "http",
                        "server": ("proxy.test", 80),
                        "path": path,
                        "query_string": b"",
                        "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
                    },
                    receive,
                )
                response = await proxy.handle_passthrough(request, base)
            else:
                response = await proxy._stream_response(
                    url=base + path,
                    headers=headers,
                    body={"model": "gpt-41-copilot"},
                    provider="openai",
                    model="gpt-41-copilot",
                    request_id="inline-auth-regression",
                    original_tokens=0,
                    optimized_tokens=0,
                    tokens_saved=0,
                    transforms_applied=[],
                    tags={},
                    optimization_latency=0.0,
                )

            assert response.status_code == 200
            if isinstance(response, StreamingResponse):
                received = b"".join([chunk async for chunk in response.body_iterator])
                if response.background is not None:
                    await response.background()
            else:
                received = response.body
            assert received == content
    finally:
        proxy._compression_executor.shutdown()
        proxy._background_compression_executor.shutdown()
