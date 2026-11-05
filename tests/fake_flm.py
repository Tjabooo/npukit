#!/usr/bin/env python3
"""Stand-in for `flm serve` so the gateway can be tested without an NPU.

Behaves like FastFlowLM where it matters: prints a load error and keeps serving when a chat
model, Whisper and the embedding model are requested together (the NPU has room for two),
returns errors as HTTP 200 + {"error": ...}, and reports decoding_speed_tps in usage.
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument("cmd")
p.add_argument("model", nargs="?")
p.add_argument("--asr", default="0")
p.add_argument("--embed", default="0")
p.add_argument("--port", type=int)
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--pmode", default="performance")
p.add_argument("--quiet", action="store_true")
a, _ = p.parse_known_args()

asr, emb = a.asr == "1", a.embed == "1"
chat_ok = bool(a.model)
if a.model and asr and emb:
    print("[ERROR]  Failed to load model: DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22)", flush=True)
    chat_ok = False
print(f"[FLM]  Starting server on port {a.port}...", flush=True)


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": a.model or "none"}], "pmode": a.pmode})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        if self.path == "/v1/chat/completions":
            req = json.loads(raw)
            if not chat_ok:
                return self._send({"error": f"Failed to load {a.model} model!"})
            return self._send({"choices": [{"message": {"role": "assistant",
                                                        "content": f"model={req['model']} pmode={a.pmode}"}}],
                               "usage": {"completion_tokens": 3, "decoding_speed_tps": 12.5}})
        if self.path == "/v1/embeddings":
            if not emb:
                return self._send({"error": "embedding model not loaded"})
            req = json.loads(raw)
            inp = req["input"] if isinstance(req["input"], list) else [req["input"]]
            return self._send({"object": "list", "data": [
                {"index": i, "embedding": [float(len(t) % 7), 1.0, float("rust" in t)]}
                for i, t in enumerate(inp)]})
        if self.path == "/v1/audio/transcriptions":
            if not asr:
                return self._send({"error": "asr model not loaded"})
            return self._send({"text": f" heard {len(raw)} bytes"})
        self._send({"error": "unknown"})


ThreadingHTTPServer((a.host, a.port), H).serve_forever()
sys.exit(0)
