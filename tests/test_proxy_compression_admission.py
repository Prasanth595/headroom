"""Whole-history admission and compression passthrough behavior."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.mark.parametrize("raw", ["0", "-1", "not-int", "1.5"])
def test_invalid_request_body_ceiling_is_rejected(raw: str) -> None:
    from headroom.proxy.request_limit_policy import resolve_request_body_max_bytes

    with pytest.raises(ValueError):
        resolve_request_body_max_bytes(raw)


def test_startup_override_changes_http_admission_after_restart() -> None:
    program = textwrap.dedent(
        """
        import json
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient
        from headroom.proxy.request_body_limit import RequestBodyLimitMiddleware

        async def preprocess(request):
            await request.json()
            return JSONResponse({"preprocessed": True})

        app = Starlette(routes=[Route("/v1/compress", preprocess, methods=["POST"])])
        app.add_middleware(RequestBodyLimitMiddleware)
        with TestClient(app) as client:
            response = client.post(
                "/v1/compress",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "a" * 1500}]},
            )
            print(json.dumps({"status": response.status_code, "body": response.json()}))
        """
    )
    results = []
    for ceiling in (1024, 2048):
        process = subprocess.run(
            [sys.executable, "-c", program],
            env={
                **os.environ,
                "HEADROOM_REQUEST_BODY_MAX_BYTES": str(ceiling),
                "DO_NOT_TRACK": "1",
                "HEADROOM_TELEMETRY": "off",
            },
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        results.append(json.loads(process.stdout))

    assert results[0]["status"] == 413
    assert results[0]["body"]["error"]["code"] == "request_too_large"
    assert results[1] == {"status": 200, "body": {"preprocessed": True}}


@pytest.fixture
def admitted_proxy(monkeypatch):
    from fastapi.testclient import TestClient

    from headroom.proxy import request_limit_policy
    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setattr(request_limit_policy, "COMPRESSION_BODY_MAX_BYTES", 1024)

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            response = json.dumps(
                {
                    "id": "msg_admission",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 101, "output_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    config = ProxyConfig(
        anthropic_api_url=f"http://127.0.0.1:{provider.server_port}",
        openai_api_url=f"http://127.0.0.1:{provider.server_port}",
        gemini_api_url=f"http://127.0.0.1:{provider.server_port}",
        bedrock_api_url=f"http://127.0.0.1:{provider.server_port}",
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        stateless=True,
        disable_kompress=True,
    )
    try:
        with TestClient(
            create_app(config),
            base_url="http://127.0.0.1",
            client=("127.0.0.1", 50000),
        ) as client:
            yield client
    finally:
        provider.shutdown()
        provider.server_close()
        thread.join()


def test_gzip_history_above_soft_threshold_stays_live_without_claiming_savings(admitted_proxy):
    raw = json.dumps(
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "history " * 3000}],
        }
    ).encode()
    encoded = gzip.compress(raw)
    assert len(encoded) < 1024 < len(raw)
    response = admitted_proxy.post(
        "/v1/messages",
        content=encoded,
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    )
    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "ok"}]
    stats = admitted_proxy.get("/stats").json()
    assert stats["tokens"]["input_provider_reported"] == 101
    assert stats["tokens"]["output"] == 1
    assert stats["tokens"]["proxy_attempted_tokens"] == 0
    assert stats["tokens"]["saved"] == 0
    assert admitted_proxy.get("/livez").status_code == 200
    following = admitted_proxy.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "next turn"}],
        },
    )
    assert following.status_code == 200
    assert admitted_proxy.get("/stats").json()["requests"]["failed"] == 0


@pytest.mark.parametrize(
    "path",
    [
        "/v1/messages",
        "/v1/chat/completions",
        "/v1/responses",
        "/v1beta/models/gemini-2.5-flash:generateContent",
        "/model/anthropic.claude-v2/invoke",
    ],
)
def test_soft_passthrough_still_obeys_decompressed_hard_ceiling(admitted_proxy, monkeypatch, path):
    from headroom.proxy import helpers

    monkeypatch.setattr(helpers, "MAX_DECOMPRESSED_BODY_SIZE", 8192)
    monkeypatch.setattr(helpers, "MAX_REQUEST_BODY_SIZE", 8192)
    content = "history " * 3000
    body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": content}],
    }
    if path == "/v1/responses":
        body = {"model": "gpt-4o", "input": content}
    elif "/models/" in path:
        body = {"contents": [{"role": "user", "parts": [{"text": content}]}]}
    elif "/model/" in path:
        body["anthropic_version"] = "bedrock-2023-05-31"
    raw = json.dumps(body).encode()
    response = admitted_proxy.post(
        path,
        content=gzip.compress(raw),
        headers={"content-type": "application/json", "content-encoding": "gzip"},
    )
    assert response.status_code == 413
    assert admitted_proxy.get("/livez").status_code == 200
