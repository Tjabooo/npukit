"""Tiny stdlib HTTP client for the gateway (and any OpenAI-compatible server)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid

from . import config as C


class ApiError(RuntimeError):
    pass


def call(path: str, payload=None, *, base: str | None = None, timeout: float = 600,
         method: str | None = None) -> dict:
    url = (base or C.base_url()) + path
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data is not None else "GET"),
                                 headers={"Content-Type": "application/json"})
    return _send(req, timeout)


def _send(req, timeout) -> dict:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise _http_error(e) from None
    except urllib.error.URLError as e:
        raise ApiError(f"gateway not reachable at {req.full_url}: {e.reason}") from None
    # FastFlowLM reports some failures as HTTP 200 with {"error": ...}
    if isinstance(out, dict) and "error" in out and not ({"data", "choices", "text"} & out.keys()):
        err = out["error"]
        raise ApiError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
    return out


def _http_error(e) -> ApiError:
    body = e.read().decode("utf-8", "replace")
    try:
        err = json.loads(body).get("error", body)
        if isinstance(err, dict):
            err = err.get("message", err)
    except (ValueError, AttributeError):
        err = body
    return ApiError(f"{e.code}: {err}")


def transcribe(wav: bytes, *, language: str = "", prompt: str = "", timeout: float = 300) -> str:
    boundary = uuid.uuid4().hex
    parts = []
    fields = {"model": "whisper-v3:turbo", "response_format": "json"}
    if language:
        fields["language"] = language
    if prompt:
        fields["prompt"] = prompt
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n".encode() + wav + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(C.base_url() + "/v1/audio/transcriptions", data=b"".join(parts),
                                 method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return _send(req, timeout).get("text", "").strip()


def chat(messages: list, model: str | None = None, *, max_tokens: int = 1024, base: str | None = None,
         temperature: float | None = None, timeout: float = 600, **extra) -> dict:
    body = {"model": model or C.load()["models"]["default"], "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    body.update(extra)
    return call("/v1/chat/completions", body, base=base, timeout=timeout)


def embed(texts: list[str], timeout: float = 600) -> list[list[float]]:
    out = call("/v1/embeddings", {"model": "embed-gemma:300m", "input": texts}, timeout=timeout)
    data = sorted(out["data"], key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]
