#!/usr/bin/env python3
"""Stand-in for llama-server and whisper-server so the llamacpp backend can be tested anywhere.

Acts as whisper-server when started with --inference-path (answers there with {"text": ...}),
otherwise as llama-server: /health, /v1/chat/completions with llama.cpp's "timings", and
/v1/embeddings when started with --embeddings. Replies name the model and the process id so
tests can tell which server answered and whether it was restarted.
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument("-m", "--model")
p.add_argument("--mmproj")
p.add_argument("-hf")
p.add_argument("--alias")
p.add_argument("--port", type=int)
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--inference-path")
p.add_argument("--embeddings", action="store_true")
a, _ = p.parse_known_args()
if a.model and not os.path.exists(a.model):
    print(f"error: failed to open model file {a.model}", flush=True)
    raise SystemExit(1)
print(f"main: server is listening on http://{a.host}:{a.port}", flush=True)


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send({"status": "ok"})
        self._send({"error": {"message": "File Not Found"}}, 404)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if a.inference_path:
            if self.path != a.inference_path:
                return self._send({"error": "not found"}, 404)
            return self._send({"text": f" whisper {os.getpid()} heard {len(raw)} bytes"})
        if self.path == "/v1/chat/completions":
            json.loads(raw)
            return self._send({"choices": [{"message": {"role": "assistant",
                                                        "content": f"alias={a.alias} pid={os.getpid()} "
                                                                   f"vision={bool(a.mmproj)}"}}],
                               "usage": {"completion_tokens": 3},
                               "timings": {"predicted_per_second": 88.5}})
        if self.path == "/v1/embeddings" and a.embeddings:
            req = json.loads(raw)
            inp = req["input"] if isinstance(req["input"], list) else [req["input"]]
            return self._send({"object": "list", "data": [
                {"index": i, "embedding": [float(len(t) % 7), 1.0, float("rust" in t)]}
                for i, t in enumerate(inp)]})
        self._send({"error": {"message": "File Not Found"}}, 404)


ThreadingHTTPServer((a.host, a.port), H).serve_forever()
