"""npud: OpenAI-compatible gateway that owns the model servers (FastFlowLM on the NPU, or
llama.cpp + whisper.cpp on a GPU).

Why a gateway: an XDNA2 NPU only has room for one chat model plus one helper (Whisper or the
embedding model), and a second flm process can't get an NPU context while a chat model is loaded.
So exactly one `flm serve` runs, started on demand with the components the last few requests
needed, stopped after an idle timeout (shorter on battery), and restarted when a request needs
something it doesn't have. Idle cost is zero: with nothing loaded the gateway blocks in epoll,
and when started by systemd socket activation it exits after a minute without a model.

On machines without an NPU (`[server] backend = "llamacpp"`) the same gateway runs one
llama-server per chat/embedding model and a whisper-server for speech, each on its own private
port, loaded on demand and unloaded on idle the same way, so the tools work unchanged.
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
TPS = re.compile(rb'"(?:decoding_speed_tps|predicted_per_second)"\s*:\s*([0-9.]+)')  # flm | llama.cpp
LLAMA_ERR = re.compile(r"^(error|E |ERROR)|\b(error|failed to|unable to)\b.*(load|open|model|alloc)", re.I)
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


class Proc:
    """One model server process: the flm child, or one llama-server / whisper-server."""

    def __init__(self, kind: str, components: list[str], chat_model: str | None, port: int,
                 argv: list[str], probe: str = "/v1/models"):
        self.kind = kind              # flm | llama | whisper
        self.components = components
        self.chat_model = chat_model
        self.port = port
        self.argv = argv
        self.probe = probe            # GET path that answers 200 once the model is loaded
        self.timeout: float | None = None  # load timeout; None = [server] start_timeout
        self.note = ""                # e.g. "CPU" when Whisper had to leave the GPU to the reserve
        self.proc: asyncio.subprocess.Process | None = None
        self.lines: collections.deque[str] = collections.deque(maxlen=60)
        self.failed = ""

    def same(self, other: "Proc") -> bool:
        return self.argv == other.argv and self.port == other.port

    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None


class Child:
    """Everything that is loaded: one flm process (all components), or a server per component."""

    def __init__(self, procs: list[Proc], pmode: str | None):
        self.procs = procs
        self.pmode = pmode
        self.started = time.time()

    @property
    def components(self) -> list[str]:
        return [x for p in self.procs for x in p.components]

    @property
    def chat_model(self) -> str | None:
        p = self.proc_for("chat")
        return p.chat_model if p else None

    @property
    def port(self) -> int:
        p = self.proc_for("chat") or self.procs[0]
        return p.port

    @property
    def proc(self):
        return self.procs[0].proc if self.procs else None

    def proc_for(self, need: str) -> Proc | None:
        return next((p for p in self.procs if need in p.components), None)

    def has(self, need: str, model: str | None) -> bool:
        p = self.proc_for(need)
        if not p or not p.alive():
            return False
        return need != "chat" or p.chat_model == model

    def alive(self) -> bool:
        return bool(self.procs) and all(p.alive() for p in self.procs)


class FlmEngine:
    """FastFlowLM: one process on the NPU serves the chat model and the helpers together."""
    name, incremental, uses_pmode = "flm", False, True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.max_components = int(cfg["server"]["max_components"])

    def check(self, need: str, model: str | None, child: "Child | None" = None) -> None:
        pass  # flm reports unknown tags itself

    def plan(self, wanted: list[str], chat_model: str | None, pmode: str) -> list[Proc]:
        s = self.cfg["server"]
        args = [os.path.expanduser(s["flm"]), "serve"]
        if "chat" in wanted:
            args.append(chat_model)
        if "asr" in wanted:
            args += ["--asr", "1"]
        if "embed" in wanted:
            args += ["--embed", "1"]
        # --quiet: flm otherwise logs every request body; load errors are still printed
        args += ["--port", str(s["flm_port"]), "--host", "127.0.0.1", "--pmode", pmode, "--quiet"]
        args += [str(a) for a in s["flm_args"]]
        return [Proc("flm", list(wanted), chat_model if "chat" in wanted else None, int(s["flm_port"]), args)]

    async def installed(self) -> list[dict]:
        p = await asyncio.create_subprocess_exec(
            os.path.expanduser(self.cfg["server"]["flm"]), "list", "--json", "--filter", "installed", "--quiet",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(p.communicate(), 20)
        data = json.loads(out)
        return [{"id": m["name"], "object": "model", "owned_by": "FastFlowLM",
                 "labels": m.get("label", []), "size_gb": m.get("footprint")}
                for m in (data.get("models", data) if isinstance(data, dict) else data)]


class LlamaCppEngine:
    """llama.cpp + whisper.cpp: a server per component (GPU or CPU), so a GPU can hold the chat
    model, Whisper and the embedding model at once and swapping one leaves the others loaded."""
    name, incremental, uses_pmode = "llamacpp", True, False
    OFFSET = {"chat": 0, "asr": 1, "embed": 2}

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lc = cfg["llamacpp"]
        self.max_components = int(self.lc.get("max_components", 3))

    def spec(self, tag: str) -> dict:
        """[llamacpp.models] entry for a tag; a bare .gguf path or a user/repo[:quant] works too."""
        spec = self.lc.get("models", {}).get(tag)
        if isinstance(spec, str):
            spec = {"path": spec} if spec.endswith(".gguf") else {"hf": spec}
        if spec is None:
            if tag.endswith(".gguf"):
                spec = {"path": tag}
            elif "/" in tag:
                spec = {"hf": tag}
            else:
                raise RuntimeError(f"unknown model {tag!r}: add it under [llamacpp.models] "
                                   f"(path = \"….gguf\" or hf = \"user/repo:quant\")")
        if not (spec.get("path") or spec.get("hf")):
            raise RuntimeError(f"[llamacpp.models] {tag!r} needs path or hf")
        for key in ("path", "mmproj"):
            if spec.get(key) and not C.expand(spec[key]).exists():
                raise RuntimeError(f"model file missing for {tag}: {spec[key]}")
        return spec

    def check(self, need: str, model: str | None, child: "Child | None" = None) -> None:
        if need == "chat":
            self.fits(model or "", self.spec(model or ""), child)
        elif need == "embed":
            self.spec(self.lc["embed_model"])
        elif need == "asr" and not C.expand(self.lc["whisper_model"]).exists():
            raise RuntimeError(f"Whisper model missing: {self.lc['whisper_model']} "
                               "(scripts/get-models.sh downloads it)")

    # -- headroom: keep [llamacpp] vram_reserve of the GPU and ram_reserve_gb of RAM for other apps
    def vram_reserve(self, vram: dict | None) -> int:
        """Bytes of VRAM to leave free, from [llamacpp] vram_reserve (a fraction of the card)."""
        frac = float(self.lc.get("vram_reserve", 0) or 0)
        return int(vram["total"] * frac) if vram and frac > 0 else 0

    def footprint(self, spec: dict) -> int | None:
        """Bytes a chat server needs (model + projector files + [llamacpp] overhead_gb); None if unknown."""
        if not spec.get("path"):
            return int(float(spec["size_gb"]) * 1e9) if spec.get("size_gb") else None
        files = [C.expand(f) for f in (spec.get("path"), spec.get("mmproj")) if f]
        return sum(f.stat().st_size for f in files) + int(float(self.lc.get("overhead_gb", 1.0)) * 1e9)

    def fits(self, tag: str, spec: dict, child: "Child | None") -> None:
        """Refuse a model whose share outside the GPU (whatever doesn't fit above the VRAM reserve)
        would leave less than ram_reserve_gb of RAM available. The chat server it replaces counts as freed."""
        keep = float(self.lc.get("ram_reserve_gb", 0) or 0) * 1e9
        need, avail = self.footprint(spec), hw.mem_available()
        if not keep or need is None or avail is None:
            return
        vram = hw.gpu_vram()
        old = child.proc_for("chat") if child else None
        freed_vram = 0
        if old and old.alive() and old.proc:
            freed_vram = hw.proc_vram(old.proc.pid)
            avail += hw.proc_rss_anon(old.proc.pid)
        room = max(0, vram["total"] - vram["used"] + freed_vram - self.vram_reserve(vram)) if vram else 0
        in_ram = max(0, need - room)
        if avail - in_ram < keep:
            gb = 1e9
            raise RuntimeError(
                f"not enough memory for {tag} right now: ~{in_ram / gb:.1f} GB would go to RAM "
                f"(GPU room {room / gb:.1f} GB after keeping {self.vram_reserve(vram) / gb:.1f} GB free), "
                f"{avail / gb:.1f} GB available, keeping {keep / gb:.0f} GB free. Close something or use a smaller model")

    def _fit_args(self) -> list[str]:
        """--fit-target = the VRAM reserve: llama.cpp puts what doesn't fit above it in RAM (MoE experts first)."""
        reserve = self.vram_reserve(hw.gpu_vram())
        args = ["--fit", "on", "--fit-target", str(reserve // 1048576)] if reserve else []
        threads = int(self.lc.get("threads", 0) or 0)
        return args + (["-t", str(threads)] if threads else [])

    def _whisper_on_gpu(self) -> bool:
        vram = hw.gpu_vram()
        if not vram:
            return True
        need = C.expand(self.lc["whisper_model"]).stat().st_size + 400 * 1048576  # + work buffers
        return vram["total"] - vram["used"] - need >= self.vram_reserve(vram)

    def _model_args(self, spec: dict) -> list[str]:
        if spec.get("path"):
            args = ["-m", str(C.expand(spec["path"]))]
            if spec.get("mmproj"):
                args += ["--mmproj", str(C.expand(spec["mmproj"]))]
            return args
        return ["-hf", spec["hf"]]

    def plan(self, wanted: list[str], chat_model: str | None, pmode: str | None) -> list[Proc]:
        base = int(self.cfg["server"]["flm_port"])
        procs = []
        for comp in wanted:
            port = base + self.OFFSET[comp]
            if comp == "asr":
                argv = [os.path.expanduser(self.lc["whisper"]), "-m", str(C.expand(self.lc["whisper_model"])),
                        "--host", "127.0.0.1", "--port", str(port),
                        "--inference-path", "/v1/audio/transcriptions", *map(str, self.lc["whisper_args"])]
                gpu = self._whisper_on_gpu()
                if not gpu:  # the GPU is at its reserve: transcribe on the CPU instead of eating into it
                    argv.append("-ng")
                    if int(self.lc.get("threads", 0) or 0):
                        argv += ["-t", str(int(self.lc["threads"]))]
                p = Proc("whisper", ["asr"], None, port, argv, "/health")
                p.note = "" if gpu else "CPU"
                procs.append(p)
                continue
            tag = chat_model if comp == "chat" else self.lc["embed_model"]
            spec = self.spec(tag or "")
            extra = self.lc["args"] if comp == "chat" else self.lc["embed_args"]
            argv = [os.path.expanduser(self.lc["server"]), *self._model_args(spec),
                    "--host", "127.0.0.1", "--port", str(port), "--alias", tag, *self._fit_args(),
                    *map(str, extra), *map(str, spec.get("args", []))]
            p = Proc("llama", [comp], tag if comp == "chat" else None, port, argv, "/health")
            if spec.get("hf") and not spec.get("path"):  # the first start downloads the model
                p.timeout = float(self.lc.get("download_timeout", 3600))
            procs.append(p)
        return procs

    async def installed(self) -> list[dict]:
        out = []
        for tag, spec in self.lc.get("models", {}).items():
            spec = {"path": spec} if isinstance(spec, str) and spec.endswith(".gguf") else \
                ({"hf": spec} if isinstance(spec, str) else spec)
            path = C.expand(spec["path"]) if spec.get("path") else None
            out.append({"id": tag, "object": "model", "owned_by": "llama.cpp",
                        "source": str(path) if path else "hf:" + str(spec.get("hf")),
                        "present": bool(path and path.exists()) if path else None,
                        "size_gb": round(path.stat().st_size / 1e9, 1) if path and path.exists() else None})
        return out


def make_engine(cfg: dict):
    return LlamaCppEngine(cfg) if C.backend(cfg) == "llamacpp" else FlmEngine(cfg)


class Gateway:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.engine = make_engine(cfg)
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

    # ---------------------------------------------------------------- model server processes
    async def _pump(self, p: Proc):
        """Read a server's (often chatty) output: keep a short tail, log errors, never block it."""
        buf = b""
        assert p.proc and p.proc.stdout
        while True:
            chunk = await p.proc.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            if len(buf) > 65536:  # request bodies with base64 images come as one huge line
                buf = b""
            for raw in lines:
                line = ANSI.sub("", raw[:400].decode("utf-8", "replace")).strip()
                if p.kind == "flm":
                    if not line.startswith(("[FLM]", "[ERROR]", "[Error]", "[Warning]")):
                        continue  # flm echoes request content even with --quiet: never parse that
                    p.lines.append(line)
                    if line.startswith(("[ERROR]", "[Error]")):
                        log("flm:", line)
                        if "Failed to load" in line or "HWCTX" in line:
                            p.failed = line
                    continue
                # llama.cpp / whisper.cpp: transcripts start with "[" (timestamps): keep them out of
                # the tail and the journal; errors look like "error: …", "E …" or "…: failed to …"
                if not line or line.startswith("["):
                    continue
                p.lines.append(line)
                if LLAMA_ERR.search(line):
                    log(f"{p.kind}:", line)

    async def _spawn(self, p: Proc) -> bool:
        s = self.cfg["server"]
        log("start:", " ".join(p.argv))
        t0 = time.monotonic()
        try:
            p.proc = await asyncio.create_subprocess_exec(
                *p.argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        except FileNotFoundError:
            hint = {"flm": "install FastFlowLM", "llama": "install llama.cpp (Arch: llama-cpp ggml-vulkan)",
                     "whisper": "install whisper.cpp (Arch: whisper-cpp)"}[p.kind]
            p.failed = f"{os.path.basename(p.argv[0])} not found: {hint}"
            return False
        except OSError as e:
            p.failed = f"{p.argv[0]}: {e.strerror or e}"
            return False
        asyncio.get_running_loop().create_task(self._pump(p))
        deadline = t0 + float(p.timeout or s["start_timeout"])
        while time.monotonic() < deadline:  # polls only while a model is loading
            if not p.alive():
                break
            if await self._probe(p.port, p.probe):
                break
            await asyncio.sleep(0.15)
        await asyncio.sleep(0.05)  # let the pump catch a load error printed just before listen
        if not p.alive() or p.failed or time.monotonic() >= deadline:
            p.failed = p.failed or (p.lines[-1] if p.lines else f"{p.kind} did not start")
            log("start failed:", p.failed)
            await self._kill(p)
            return False
        log(f"ready in {time.monotonic() - t0:.1f}s: {p.components} {p.chat_model or ''}")
        return True

    async def _load(self, wanted: list[str], chat_model: str | None, pmode: str | None) -> list[str]:
        """Make `wanted` loaded: flm restarts its one process; llama.cpp starts the servers that are
        missing or must change and leaves the rest running. Returns the components that failed."""
        plan = self.engine.plan(wanted, chat_model, pmode)
        old = self.child.procs if self.child else []
        keep = [p for p in old if p.alive() and any(p.same(q) for q in plan)] if self.engine.incremental else []
        drop = [p for p in old if p not in keep]
        new = [q for q in plan if not any(q.same(p) for p in keep)]
        if drop and self.inflight:  # never pull a model out from under a running request
            self.idle.clear()
            await self.idle.wait()
        for p in drop:
            await self._kill(p)
        if drop and self.engine.name == "flm":
            await asyncio.sleep(0.3)  # the driver frees the NPU context right after exit
        self.child = Child(keep + new, pmode if self.engine.uses_pmode else None)
        if not new:
            return []
        self.state = "loading"
        self.publish()
        ok = await asyncio.gather(*(self._spawn(q) for q in new))
        failed = [q for q, good in zip(new, ok) if not good]
        self.child.procs = [p for p in self.child.procs if p not in failed]
        if failed:
            self.last_error = failed[0].failed
        if self.child.procs:
            self.loaded_at = time.time()
            self.state = "ready"
        else:
            self.child = None
            self.state = "error"
        self.publish()
        if failed and not self.child:
            self.arm_exit()
        return [x for q in failed for x in q.components]

    async def _probe(self, port: int, path: str = "/v1/models") -> bool:
        try:
            r, w = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 1)
        except (OSError, asyncio.TimeoutError):
            return False
        try:
            w.write(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n".encode())
            await w.drain()
            line = await asyncio.wait_for(r.readline(), 5)
            return b" 200" in line
        except (OSError, asyncio.TimeoutError):
            return False
        finally:
            w.close()

    async def _kill(self, p: Proc):
        if not p.proc or not p.alive():
            return
        log("stop:", p.components, p.chat_model or "")
        p.proc.terminate()
        try:
            await asyncio.wait_for(p.proc.wait(), 8)
        except asyncio.TimeoutError:
            log(f"{p.kind} ignored SIGTERM, killing")
            p.proc.kill()
            await p.proc.wait()

    async def _stop(self):
        child, self.child = self.child, None
        if self.unload_handle:
            self.unload_handle.cancel()
            self.unload_handle = None
        if not child:
            return
        procs = [p for p in child.procs if p.alive()]
        for p in procs:
            await self._kill(p)
        if procs and self.engine.name == "flm":
            await asyncio.sleep(0.3)  # the driver frees the hardware context right after exit
        self.state = "off"
        self.publish()
        self.arm_exit()

    async def acquire(self, need: str, model: str | None = None, pmode: str | None = None) -> int:
        """Make sure something serves `need`, count the caller as in flight, return its port."""
        if not self.engine.uses_pmode:
            pmode = None
        async with self.lock:
            c = self.child
            if not (c and c.has(need, model) and (pmode is None or c.pmode == pmode)):
                self.engine.check(need, model, c)  # unknown model, missing file, no room: fail before unloading
                now, timeout = time.monotonic(), self.idle_timeout()
                keep = []
                if c:
                    keep = [x for x in c.components if x != need and now - self.last_used.get(x, 0) < timeout
                            and (c.proc_for(x) and c.proc_for(x).alive())]
                    keep.sort(key=lambda x: -self.last_used.get(x, 0))
                wanted = ([need] + keep)[: self.engine.max_components]
                chat_model = model if need == "chat" else (c.chat_model if c and "chat" in wanted else None)
                pm = (pmode or self.pmode()) if self.engine.uses_pmode else None
                failed = await self._load(wanted, chat_model, pm)
                if need in failed and len(wanted) > 1 and not self.engine.incremental:
                    # flm: the NPU may not fit the combination; try the one component that is needed
                    failed = await self._load([need], model if need == "chat" else None, pm)
                if need in failed or not (self.child and self.child.proc_for(need)):
                    raise RuntimeError(self.last_error or f"{self.engine.name} failed to start")
            self.last_used[need] = time.monotonic()
            self.inflight += 1
            if self.unload_handle:
                self.unload_handle.cancel()
                self.unload_handle = None
            self.publish()
            return self.child.proc_for(need).port

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
            "pmode": (c.pmode if c else self.pmode()) if self.engine.uses_pmode else None,
            "backend": self.engine.name,
            "runs_on": C.engine_label(self.cfg),
            "power": "ac" if ac else "battery",
            "idle_unload": self.idle_timeout(),
            "inflight": self.inflight,
            "loaded_since": self.loaded_at if c else None,
            "flm_pid": c.proc.pid if c and c.proc else None,
            "procs": [{"kind": p.kind, "components": p.components, "model": p.chat_model, "port": p.port,
                       "pid": p.proc.pid if p.proc else None, "note": p.note,
                       "vram_mb": hw.proc_vram(p.proc.pid) // 1048576 if p.proc else 0} for p in c.procs] if c else [],
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
        where = C.engine_label(self.cfg)
        tip = [f"{where} · {st}" + (f" · {act}" if act else "")]
        pm = self.engine.uses_pmode
        if c:
            names = {"asr": "whisper (speech to text)", "embed": "embeddings"}
            tip.append("loaded: " + ", ".join(
                (p.chat_model if x == "chat" else names[x]) + (f" on the {p.note}" if p.note else "")
                for p in c.procs for x in p.components))
            tip.append((f"pmode {c.pmode} · " if pm else "") + f"unloads after {int(self.idle_timeout())} s idle "
                       f"({'AC' if ac else 'battery'})")
        else:
            tip.append(f"nothing loaded · {'AC' if ac else 'battery'}" + (f" → pmode {self.pmode()}" if pm else ""))
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
        # machine-readable twin for other tools (e.g. a fleet dashboard reading it over SSH)
        st = {"state": st, "activity": act, "components": c.components if c else [],
              "chat_model": c.chat_model if c else None,
              "pmode": (c.pmode if c else self.pmode()) if pm else None, "runs_on": where,
              "power": "ac" if ac else "battery", "last_tps": self.last_tps if c else None,
              "updated": time.time()}
        try:
            sp = path.with_name("status.json")
            tmp = sp.with_suffix(".tmp")
            tmp.write_text(json.dumps(st) + "\n")
            os.replace(tmp, sp)
        except OSError as e:
            log("status write failed:", e)
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
                self.models_cache = (now, await self.engine.installed())
            except (OSError, ValueError, KeyError, asyncio.TimeoutError):
                pass
        ids = [{"id": a, "object": "model", "owned_by": "npukit", "alias_of": C.resolve_model(a, self.cfg)}
               for a in self.cfg["models"]["aliases"]]
        return {"object": "list", "data": ids + list(self.models_cache[1])}

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
