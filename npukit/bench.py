"""`npu bench`: tokens/s and power for NPU models (via the gateway) and, for comparison, any
model on another OpenAI-compatible server (e.g. Lemonade's llama.cpp on the iGPU or CPU).

Power comes from sysfs, no root needed: amdgpu's PPT sensor is the whole APU package on
Ryzen AI laptops, and the battery's power_now is the full system draw while discharging.
"""

from __future__ import annotations

import json
import statistics
import threading
import time

from . import client
from . import config as C
from . import hw

PROMPT = ("Explain how a hash map works, including hashing, collisions (chaining vs open addressing), "
          "load factor and resizing. Use about 300 words.")


class Sampler(threading.Thread):
    def __init__(self, hz: float = 5):
        super().__init__(daemon=True)
        self.ppt, self.bat = hw.ppt_path(), hw.battery_dir()
        self.dt = 1 / hz
        self.samples: list[tuple[float | None, float | None]] = []
        self.go = threading.Event()
        self.go.set()

    def run(self):
        while self.go.is_set():
            self.samples.append((hw.watts(self.ppt), hw.battery_watts(self.bat)))
            time.sleep(self.dt)

    def stop(self) -> dict:
        self.go.clear()
        self.join()
        ppt = [p for p, _ in self.samples if p is not None]
        bat = [b for _, b in self.samples if b]
        return {"ppt_w": statistics.mean(ppt) if ppt else None,
                "ppt_peak_w": max(ppt) if ppt else None,
                "bat_w": statistics.mean(bat) if bat else None}


def idle_power(seconds: float = 4) -> dict:
    s = Sampler()
    s.start()
    time.sleep(seconds)
    return s.stop()


def run_one(model: str, base: str | None, tokens: int, extra: dict) -> dict:
    msgs = [{"role": "user", "content": PROMPT}]
    s = Sampler()
    s.start()
    t0 = time.monotonic()
    out = client.chat(msgs, model, max_tokens=tokens, base=base, temperature=0, **extra)
    wall = time.monotonic() - t0
    pw = s.stop()
    u = out.get("usage", {})
    timings = out.get("timings", {})  # llama.cpp
    n = u.get("completion_tokens") or timings.get("predicted_n") or 0
    tps = u.get("decoding_speed_tps") or timings.get("predicted_per_second")
    ttft = u.get("prefill_duration_ttft") or (timings.get("prompt_ms", 0) / 1000 or None)
    if base and not tps:  # Lemonade keeps per-request stats for every backend
        try:
            st = client.call("/api/v1/stats", base=base, timeout=10)
            tps, ttft = st.get("tokens_per_second") or tps, st.get("time_to_first_token") or ttft
        except client.ApiError:
            pass
    tps = tps or (n / wall if wall else 0)
    r = {"model": model, "tokens": n, "wall_s": wall, "tps": tps, "ttft_s": ttft, **pw}
    if pw["ppt_w"] and tps:
        r["j_per_tok"] = pw["ppt_w"] / tps
    return r


def main(args) -> int:
    cfg = C.load()
    ac = hw.on_ac(cfg["power"]["ac_online"])
    rows = []
    base_idle = idle_power()
    print(f"power source: {'AC' if ac else 'battery'} · idle package {base_idle['ppt_w'] or 0:.1f} W"
          + (f" · system {base_idle['bat_w']:.1f} W" if base_idle["bat_w"] and not ac else ""), flush=True)
    targets = [(m, None) for m in (args.models or ([] if args.other else ["fast"]))]
    if args.other:
        url = args.other_url.rstrip("/").removesuffix("/v1")
        targets += [(m, url) for m in args.other]
    pmodes = args.pmode.split(",") if args.pmode else [None]
    nothink = {"chat_template_kwargs": {"enable_thinking": False}}
    for model, base in targets:
        for pm in (pmodes if base is None else [None]):
            t0 = time.monotonic()
            if base is None:
                name = C.resolve_model(model, cfg)
                client.call("/npu/load", {"component": "chat", "model": name, "pmode": pm}, timeout=600)
                label = name + (f" [{pm}]" if pm else "")
            else:  # MODEL@backend, e.g. Qwen3.5-4B-GGUF@vulkan (Lemonade's llama.cpp backends)
                name, _, backend = model.partition("@")
                body = {"model_name": name}
                if backend:
                    body["llamacpp_backend"] = backend
                try:
                    client.call("/api/v1/load", body, base=base, timeout=900)
                except client.ApiError as e:
                    print(f"  {model}: load failed: {e}")
                    continue
                label = f"{name} [{backend or 'default'}]"
            load_s = time.monotonic() - t0
            client.chat([{"role": "user", "content": "hi"}], name, max_tokens=8, base=base, **nothink)
            runs = [run_one(name, base, args.tokens, nothink) for _ in range(args.runs)]
            best = max(runs, key=lambda r: r["tps"] or 0)
            best.update(label=label, load_s=load_s, ac=ac)
            rows.append(best)
            print(_fmt(best), flush=True)
            if base:
                try:
                    client.call("/api/v1/unload", {}, base=base, timeout=60)
                except client.ApiError:
                    pass
    if args.json:
        print(json.dumps(rows, indent=1))
    return 0


def _fmt(r: dict) -> str:
    def f(v, spec):
        return format(v, spec) if isinstance(v, (int, float)) else "-"
    return (f"  {r['label']:<40} {f(r['tps'], '6.1f')} tok/s  ttft {f(r['ttft_s'], '5.2f')} s  "
            f"pkg {f(r['ppt_w'], '5.1f')} W (peak {f(r['ppt_peak_w'], '4.1f')})  "
            + ("" if r.get("ac") else f"sys {f(r['bat_w'], '5.1f')} W  ")
            + f"{f(r.get('j_per_tok'), '5.2f')} J/tok  load {f(r['load_s'], '4.1f')} s")
