"""npud: OpenAI-compatible gateway that owns the one FastFlowLM process on the NPU.

Why a gateway: an XDNA2 NPU only has room for one chat model plus one helper (Whisper or the
embedding model), and a second flm process can't get an NPU context while a chat model is loaded.
So exactly one `flm serve` runs, started on demand with the components the last few requests
needed, stopped after an idle timeout (shorter on battery), and restarted when a request needs
something it doesn't have. Idle cost is zero: with nothing loaded the gateway blocks in epoll,
and when started by systemd socket activation it exits after a minute without a model.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import re
import signal
import socket
import sys
import time

from . import config as C
from . import hw

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TPS = re.compile(rb'"decoding_speed_tps"\s*:\s*([0-9.]+)')
ROUTES = {
    "/v1/chat/completions": "chat",
    "/v1/completions": "chat",
    "/v1/responses": "chat",
    "/v1/embeddings": "embed",
    "/v1/audio/transcriptions": "asr",
    "/v1/audio/translations": "asr",
}
COMPONENT_LABEL = {"asr": "STT", "embed": "EMB"}
ACTIVITY_LABEL = {"rec": "● REC", "stt": "STT…", "ocr": "OCR…", "index": "INDEX…"}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Child:
    def __init__(self, components: list[str], chat_model: str | None, pmode: str, port: int):
        self.components = components
        self.chat_model = chat_model
        self.pmode = pmode
        self.port = port
        self.proc: asyncio.subprocess.Process | None = None
        self.lines: collections.deque[str] = collections.deque(maxlen=60)
        self.failed = ""
        self.started = time.time()

    def has(self, need: str, model: str | None) -> bool:
        if need not in self.components:
            return False
        return need != "chat" or self.chat_model == model

    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None


class Gateway:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.child: Child | None = None
        self.state = "off"           # off | loading | ready | error
        self.activity: str | None = None
        self.inflight = 0
        self.conns = 0
        self.last_used: dict[str, float] = {}
        self.lock = asyncio.Lock()
        self.idle = asyncio.Event()
        self.idle.set()
        self.unload_handle: asyncio.TimerHandle | None = None
        self.exit_handle: asyncio.TimerHandle | None = None
        self.socket_activated = False
        self.last_tps: float | None = None
        self.last_error = ""
        self.loaded_at = 0.0
        self.models_cache: tuple[float, list] = (0.0, [])
        self.ac_path = hw.ac_online_path(cfg["power"]["ac_online"])
        self._bar_last = ""

    # ---------------------------------------------------------------- power policy
    def on_ac(self) -> bool:
        return hw.on_ac(self.ac_path)

    def idle_timeout(self) -> float:
        p = self.cfg["power"]
        return float(p["ac_idle_unload"] if self.on_ac() else p["battery_idle_unload"])

    def pmode(self) -> str:
        p = self.cfg["power"]
        return p["ac_pmode"] if self.on_ac() else p["battery_pmode"]

    # ---------------------------------------------------------------- flm process
    async def _pump(self, child: Child):
        """Read flm's (very chatty) stdout: keep a short tail, log errors, never block it."""
        buf = b""
        assert child.proc and child.proc.stdout
        while True:
            chunk = await child.proc.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            if len(buf) > 65536:  # request bodies with base64 images come as one huge line
                buf = b""
            for raw in lines:
                line = ANSI.sub("", raw[:400].decode("utf-8", "replace")).strip()
                if not line.startswith(("[FLM]", "[ERROR]", "[Error]", "[Warning]")):
                    continue  # flm echoes request content even with --quiet: never parse that
                child.lines.append(line)
                if line.startswith(("[ERROR]", "[Error]")):
                    log("flm:", line)
                    if "Failed to load" in line or "HWCTX" in line:
                        child.failed = line

    async def _start(self, components: list[str], chat_model: str | None, pmode: str) -> bool:
        s = self.cfg["server"]
        args = [os.path.expanduser(s["flm"]), "serve"]
        if "chat" in components:
            args.append(chat_model)
        if "asr" in components:
            args += ["--asr", "1"]
        if "embed" in components:
            args += ["--embed", "1"]
        # --quiet: flm otherwise logs every request body; load errors are still printed
        args += ["--port", str(s["flm_port"]), "--host", "127.0.0.1", "--pmode", pmode, "--quiet"]
        args += [str(a) for a in s["flm_args"]]
        child = Child(components, chat_model, pmode, int(s["flm_port"]))
        self.child = child
        self.state = "loading"
        self.publish()
        log("start:", " ".join(args))
        t0 = time.monotonic()
        child.proc = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        asyncio.get_running_loop().create_task(self._pump(child))
        deadline = t0 + float(s["start_timeout"])
        while time.monotonic() < deadline:  # polls only while a model is loading
            if not child.alive():
                break
            if await self._probe(child.port):
                break
            await asyncio.sleep(0.15)
        await asyncio.sleep(0.05)  # let the pump catch a load error printed just before listen
        if not child.alive() or child.failed or time.monotonic() >= deadline:
            self.last_error = child.failed or (child.lines[-1] if child.lines else "flm did not start")
            log("start failed:", self.last_error)
            await self._stop()
            self.state = "error"
            self.publish()
            return False
        self.loaded_at = time.time()
        self.state = "ready"
        log(f"ready in {time.monotonic() - t0:.1f}s: {components} {chat_model or ''} pmode={pmode}")
        self.publish()
        return True

    async def _probe(self, port: int) -> bool:
        try:
            r, w = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 1)
        except (OSError, asyncio.TimeoutError):
            return False
        try:
            w.write(b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            await w.drain()
            line = await asyncio.wait_for(r.readline(), 5)
            return b" 200" in line
        except (OSError, asyncio.TimeoutError):
            return False
        finally:
            w.close()

    async def _stop(self):
        child, self.child = self.child, None
        if self.unload_handle:
            self.unload_handle.cancel()
            self.unload_handle = None
        if not child or not child.proc:
            return
        if child.alive():
            log("stop:", child.components, child.chat_model or "")
            child.proc.terminate()
            try:
                await asyncio.wait_for(child.proc.wait(), 8)
            except asyncio.TimeoutError:
                log("flm ignored SIGTERM, killing")
                child.proc.kill()
                await child.proc.wait()
            await asyncio.sleep(0.3)  # the driver frees the hardware context right after exit
        self.state = "off"
        self.publish()
        self.arm_exit()

    async def acquire(self, need: str, model: str | None = None, pmode: str | None = None) -> int:
        """Make sure the flm child serves `need`, count the caller as in flight, return the port."""
        async with self.lock:
            c = self.child
            if not (c and c.alive() and c.has(need, model) and (pmode is None or c.pmode == pmode)):
                now, timeout = time.monotonic(), self.idle_timeout()
                keep = []
                if c and c.alive():
                    keep = [x for x in c.components if x != need and now - self.last_used.get(x, 0) < timeout]
                    keep.sort(key=lambda x: -self.last_used.get(x, 0))
                wanted = ([need] + keep)[: int(self.cfg["server"]["max_components"])]
                chat_model = model if need == "chat" else (c.chat_model if c and "chat" in wanted else None)
                if self.inflight:
                    self.idle.clear()
                    await self.idle.wait()
                await self._stop()
                pm = pmode or self.pmode()
                ok = await self._start(wanted, chat_model, pm)
                if not ok and len(wanted) > 1:
                    ok = await self._start([need], model if need == "chat" else None, pm)
                if not ok:
                    raise RuntimeError(self.last_error or "flm failed to start")
            self.last_used[need] = time.monotonic()
            self.inflight += 1
            if self.unload_handle:
                self.unload_handle.cancel()
                self.unload_handle = None
            self.publish()
            return self.child.port

    def release(self, need: str | None = None):
        self.inflight = max(0, self.inflight - 1)
        if need:
            self.last_used[need] = time.monotonic()
        if self.inflight == 0:
            self.idle.set()
            self.arm_unload()
        self.publish()

    def arm_unload(self):
        if self.unload_handle:
            self.unload_handle.cancel()
            self.unload_handle = None
        c = self.child
        if not c or self.inflight:
            return
        timeout = self.idle_timeout()
        last = max((self.last_used.get(x, 0) for x in c.components), default=0)
        delay = max(1.0, last + timeout - time.monotonic())
        loop = asyncio.get_running_loop()
        self.unload_handle = loop.call_later(delay, lambda: loop.create_task(self._idle_unload()))

    async def _idle_unload(self):
        async with self.lock:
            if self.child and not self.inflight:
                log("idle: unloading")
                await self._stop()

    def arm_exit(self):
        if self.exit_handle:
            self.exit_handle.cancel()
            self.exit_handle = None
        if not self.socket_activated or self.child or self.conns or self.activity:
            return
        loop = asyncio.get_running_loop()
        self.exit_handle = loop.call_later(float(self.cfg["server"]["exit_after"]), self._exit)

    def _exit(self):
        if self.child or self.conns or self.activity:
            return
        log("nothing loaded, exiting (systemd starts me again on the next request)")
        self.publish(force_off=True)
        os._exit(0)

    # ---------------------------------------------------------------- status / waybar
    def status(self) -> dict:
        c = self.child
        ac = self.on_ac()
        return {
            "state": "busy" if self.inflight and self.state == "ready" else self.state,
            "activity": self.activity,
            "components": c.components if c else [],
            "chat_model": c.chat_model if c else None,
            "pmode": c.pmode if c else self.pmode(),
            "power": "ac" if ac else "battery",
            "idle_unload": self.idle_timeout(),
            "inflight": self.inflight,
            "loaded_since": self.loaded_at if c else None,
            "flm_pid": c.proc.pid if c and c.proc else None,
            "last_tps": self.last_tps,
            "last_error": self.last_error,
            "npu": hw.npu_info(),
        }

    def publish(self, force_off: bool = False):
        b = self.cfg["bar"]
        col = {"tag": "#9b5de5", "dim": "#6c6c8a", "text": "#babbf1", "hot": "#f15bb5",
               "load": "#00c6ff", "rec": "#ff5252", **b.get("colors", {})}
        st = "off" if force_off else self.status()["state"]
        c = None if force_off else self.child
        act = None if force_off else self.activity

        def tag(color):
            return f"<span size='8pt' weight='800' color='{color}'>{b['tag']}</span>  "

        parts = []
        if c and c.chat_model:
            parts.append(c.chat_model.replace(":", " "))
        if c:
            parts += [COMPONENT_LABEL[x] for x in c.components if x in COMPONENT_LABEL]
        what = " + ".join(parts)
        if act == "rec":  # solid chip, colored by the bar's CSS (#custom-npu.rec)
            cls = "rec"
            text = f"<span size='8pt' weight='800'>{b['tag']}</span>  <span weight='800'>{ACTIVITY_LABEL['rec']}</span>"
        elif act:
            cls, text = "busy " + act, tag(col["hot"]) + \
                f"<span weight='800' color='{col['hot']}'>{ACTIVITY_LABEL.get(act, act.upper())}</span>"
        elif st == "off":
            cls, text = "off", f"<span size='8pt' weight='800' color='{col['dim']}'>{b['tag']}</span>"
        elif st == "loading":
            cls, text = "loading", tag(col["tag"]) + f"<span color='{col['load']}'>{what or 'loading'}</span>"
        elif st == "busy":
            cls, text = "busy", tag(col["hot"]) + f"<span weight='800' color='{col['hot']}'>{what}</span>"
        elif st == "error":
            cls, text = "error", tag(col["rec"]) + f"<span color='{col['rec']}'>error</span>"
        else:
            cls, text = "ready", tag(col["tag"]) + f"<span color='{col['text']}'>{what}</span>"
        ac = self.on_ac()
        tip = [f"NPU · {st}" + (f" · {act}" if act else "")]
        if c:
            tip.append("loaded: " + ", ".join(
                (c.chat_model if x == "chat" else {"asr": "whisper (speech to text)",
                                                   "embed": "embeddings"}[x]) for x in c.components))
            tip.append(f"pmode {c.pmode} · unloads after {int(self.idle_timeout())} s idle "
                       f"({'AC' if ac else 'battery'})")
        else:
            tip.append(f"nothing loaded · {'AC' if ac else 'battery'} → pmode {self.pmode()}")
        if self.last_tps and c:
            tip.append(f"last reply {self.last_tps:.1f} tok/s")
        if st == "error" and self.last_error:
            tip.append(self.last_error[:120])
        tip.append("click: menu · right-click: unload")
        out = json.dumps({"text": text, "class": cls.split(), "tooltip": "\n".join(tip), "alt": cls.split()[0]})
        if out == self._bar_last:
            return
        self._bar_last = out
        path = C.expand(b["file"])
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(out + "\n")
            os.replace(tmp, path)
        except OSError as e:
            log("bar write failed:", e)
        sig = int(b.get("signal") or 0)
        if sig:
            _signal_waybar(signal.SIGRTMIN + sig)

    # ---------------------------------------------------------------- HTTP
    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.conns += 1
        if self.exit_handle:
            self.exit_handle.cancel()
            self.exit_handle = None
        try:
            await self._handle(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except Exception as e:  # never let one request kill the gateway
            log("request failed:", repr(e))
            try:
                await _respond(writer, 502, {"error": {"message": str(e), "type": "npukit_error"}})
            except Exception:
                pass
        finally:
            self.conns -= 1
            try:
                writer.close()
            except Exception:
                pass
            self.arm_exit()

    async def _handle(self, reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        method, target, _ = lines[0].split(" ", 2)
        headers: list[tuple[str, str]] = []
        for ln in lines[1:]:
            if ":" in ln:
                k, v = ln.split(":", 1)
                headers.append((k.strip(), v.strip()))
        h = {k.lower(): v for k, v in headers}
        if h.get("expect", "").lower() == "100-continue":
            writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
            await writer.drain()
        body = b""
        if "chunked" in h.get("transfer-encoding", "").lower():
            body = await _read_chunked(reader)
        elif h.get("content-length"):
            body = await reader.readexactly(int(h["content-length"]))
        path = target.split("?", 1)[0].rstrip("/") or "/"

        # -- gateway's own endpoints
        if method == "GET" and path in ("/health", "/npu/status", "/api/v1/health"):
            return await _respond(writer, 200, self.status())
        if method == "GET" and path == "/v1/models":
            return await _respond(writer, 200, await self.models())
        if method == "POST" and path.startswith("/npu/"):
            req = json.loads(body or b"{}")
            return await self.control(path[5:], req, writer)

        need = ROUTES.get(path)
        model = None
        if need == "chat" and body:
            try:
                req = json.loads(body)
                model = C.resolve_model(req.get("model"), self.cfg)
                req["model"] = model
                body = json.dumps(req).encode()
            except ValueError:
                return await _respond(writer, 400, {"error": {"message": "invalid JSON"}})
        elif need is None:
            if not (self.child and self.child.alive()):
                return await _respond(writer, 404, {"error": {"message": f"unknown endpoint {path}"}})
        try:
            port = await self.acquire(need, model) if need else self.child.port
        except RuntimeError as e:
            return await _respond(writer, 503, {"error": {"message": str(e), "type": "npu_unavailable"}})
        if not need:
            self.inflight += 1
        try:
            await self._forward(port, method, target, headers, body, writer)
        finally:
            self.release(need)

    async def _forward(self, port, method, target, headers, body, writer):
        ur, uw = await asyncio.open_connection("127.0.0.1", port)
        skip = {"host", "connection", "content-length", "expect", "transfer-encoding", "keep-alive"}
        out = [f"{method} {target} HTTP/1.1", f"Host: 127.0.0.1:{port}", "Connection: close"]
        out += [f"{k}: {v}" for k, v in headers if k.lower() not in skip]
        if body or method in ("POST", "PUT", "PATCH"):
            out.append(f"Content-Length: {len(body)}")
        uw.write(("\r\n".join(out) + "\r\n\r\n").encode("latin-1") + body)
        await uw.drain()
        tail = collections.deque(maxlen=8)
        try:
            rhead = await ur.readuntil(b"\r\n\r\n")
            rlines = rhead.decode("latin-1").split("\r\n")
            rh = {}
            keep = [rlines[0]]
            for ln in rlines[1:]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    rh[k.strip().lower()] = v.strip()
                    if k.strip().lower() not in ("connection", "keep-alive"):
                        keep.append(ln)
            keep.append("Connection: close")
            writer.write(("\r\n".join(keep) + "\r\n\r\n").encode("latin-1"))
            if "chunked" in rh.get("transfer-encoding", "").lower():
                while True:
                    size_line = await ur.readline()
                    writer.write(size_line)
                    size = int(size_line.split(b";")[0].strip() or b"0", 16)
                    if size == 0:
                        while True:
                            t = await ur.readline()
                            writer.write(t)
                            if t in (b"\r\n", b"\n", b""):
                                break
                        break
                    data = await ur.readexactly(size + 2)
                    writer.write(data)
                    tail.append(data[-4096:])
                    await writer.drain()
            elif "content-length" in rh:
                left = int(rh["content-length"])
                while left > 0:
                    data = await ur.read(min(left, 65536))
                    if not data:
                        break
                    left -= len(data)
                    writer.write(data)
                    tail.append(data[-4096:])
                    await writer.drain()
            else:
                while data := await ur.read(65536):
                    writer.write(data)
                    tail.append(data[-4096:])
                    await writer.drain()
            await writer.drain()
        finally:
            uw.close()
        m = TPS.search(b"".join(tail))
        if m:
            self.last_tps = float(m.group(1))

    async def models(self) -> dict:
        now = time.monotonic()
        if now - self.models_cache[0] > 60:
            try:
                p = await asyncio.create_subprocess_exec(
                    os.path.expanduser(self.cfg["server"]["flm"]), "list", "--json", "--filter", "installed", "--quiet",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                out, _ = await asyncio.wait_for(p.communicate(), 20)
                data = json.loads(out)
                self.models_cache = (now, data.get("models", data) if isinstance(data, dict) else data)
            except (OSError, ValueError, asyncio.TimeoutError):
                pass
        ids = [{"id": a, "object": "model", "owned_by": "npukit", "alias_of": C.resolve_model(a, self.cfg)}
               for a in self.cfg["models"]["aliases"]]
        for m in self.models_cache[1]:
            ids.append({"id": m["name"], "object": "model", "owned_by": "FastFlowLM",
                        "labels": m.get("label", []), "size_gb": m.get("footprint")})
        return {"object": "list", "data": ids}

    async def control(self, action: str, req: dict, writer):
        if action == "stop":
            async with self.lock:
                if self.inflight:
                    return await _respond(writer, 409, {"error": "busy"})
                await self._stop()
            return await _respond(writer, 200, self.status())
        if action in ("load", "warm"):
            comp = req.get("component", "chat")
            model = C.resolve_model(req.get("model"), self.cfg) if comp == "chat" else None
            if req.get("activity") is not None:
                self.activity = req["activity"] or None
                self.publish()
            try:
                await self.acquire(comp, model, req.get("pmode"))
            except RuntimeError as e:
                return await _respond(writer, 503, {"error": str(e)})
            self.release(comp)
            return await _respond(writer, 200, self.status())
        if action == "activity":
            self.activity = req.get("activity") or None
            self.publish()
            self.arm_exit()
            return await _respond(writer, 200, self.status())
        if action == "release":  # caller is done with a component: let it age out now
            comp = req.get("component")
            if comp in self.last_used:
                self.last_used[comp] = 0
            self.arm_unload()
            return await _respond(writer, 200, self.status())
        return await _respond(writer, 404, {"error": f"unknown action {action}"})


def _signal_waybar(sig: int):
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                with open(f"/proc/{pid}/comm") as f:
                    if f.read().strip() in ("waybar", ".waybar-wrapped"):
                        os.kill(int(pid), sig)
            except OSError:
                pass


async def _read_chunked(reader) -> bytes:
    out = b""
    while True:
        size = int((await reader.readline()).split(b";")[0].strip() or b"0", 16)
        if size == 0:
            await reader.readline()
            return out
        out += await reader.readexactly(size)
        await reader.readline()


async def _respond(writer, code: int, obj) -> None:
    body = json.dumps(obj).encode()
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 409: "Conflict",
              502: "Bad Gateway", 503: "Service Unavailable"}.get(code, "OK")
    writer.write(f"HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\n"
                 f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
    await writer.drain()


async def serve(cfg: dict) -> None:
    gw = Gateway(cfg)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for s in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(s, stop.set)
    if os.environ.get("LISTEN_PID") == str(os.getpid()) and int(os.environ.get("LISTEN_FDS", "0")) >= 1:
        sock = socket.socket(fileno=3)
        gw.socket_activated = True
        server = await asyncio.start_server(gw.handle, sock=sock, limit=1 << 20)
        log("npud: socket-activated on", sock.getsockname())
    else:
        host, port = cfg["server"]["listen"].rsplit(":", 1)
        server = await asyncio.start_server(gw.handle, host, int(port), limit=1 << 20)
        log(f"npud: listening on {host}:{port}")
    gw.publish()
    gw.arm_exit()
    async with server:
        await stop.wait()
    async with gw.lock:
        await gw._stop()
    gw.publish(force_off=True)


def main() -> None:
    asyncio.run(serve(C.load()))
