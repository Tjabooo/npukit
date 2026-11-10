# Benchmarks

Machine: Ryzen AI 9 HX 370 (XDNA2 NPU, Radeon 890M), 27 GiB usable RAM, Linux 7.2 (in-tree `amdxdna`),
NPU firmware 1.1.2.64, FastFlowLM 1.0.12, Lemonade 2026.44.1 (llama.cpp b11393) for the iGPU/CPU rows.
`npu bench`: one warm-up, best of 2 runs, 256 new tokens, thinking off, temperature 0.

- **pkg** = APU package power (amdgpu `PPT` sensor) averaged over the generation, idle was 6–9 W.
- **J/tok** = pkg ÷ tok/s, i.e. package energy per generated token.

## On AC

| Model | Runs on | tok/s | TTFT | pkg W | J/tok | load |
|---|---|---:|---:|---:|---:|---:|
| qwen3.5:4b (powersaver) | NPU | 15.8 | 0.77 s | 15.8 | 1.00 | 4.0 s |
| qwen3.5:4b (balanced) | NPU | 15.8 | 0.73 s | 15.7 | 1.00 | 4.5 s |
| qwen3.5:4b (performance) | NPU | 15.8 | 0.74 s | 15.6 | 0.99 | 3.6 s |
| gpt-oss:20b | NPU | 18.8 | 3.81 s | 17.6 | 0.94 | 21.8 s |
| qwen3.5:9b | NPU | 9.5 | 1.12 s | 15.8 | 1.67 | 11.0 s |
| Qwen3.5-4B Q4_K_M GGUF | iGPU, Vulkan | 24.7 | 0.63 s | 23.7 | 0.96 | 5.6 s |
| Qwen3.5-4B Q4_K_M GGUF | iGPU, ROCm | 23.9 | 0.37 s | 21.2 | 0.88 | 4.4 s |
| Qwen3.5-4B Q4_K_M GGUF | CPU | 15.1 | 0.51 s | 20.8 | 1.37 | 3.3 s |

What this says:

- `--pmode` makes no measurable difference to decode speed or power for qwen3.5:4b on this machine.
- On AC the iGPU is ~55 % faster than the NPU at about the same energy per token. The NPU's case is that it
  leaves the iGPU and CPU free (FastFlowLM uses ~18 % of one core while generating, 0 % when idle), not raw efficiency.
- gpt-oss:20b (MoE, ~3.6B active) decodes faster than the dense 9B and is the better "quality" model, but
  takes ~22 s to load and ~4 s to the first token.

## On battery

The laptop's firmware caps the package at ~12 W on battery (platform profile `quiet`), so every backend runs
into the same power ceiling and energy per token follows speed.

| Model | Runs on | tok/s | TTFT | pkg W | J/tok | load |
|---|---|---:|---:|---:|---:|---:|
| qwen3.5:4b (powersaver) | NPU | 15.8 | 0.84 s | 11.4 | 0.72 | 3.5 s |
| gpt-oss:20b (powersaver) | NPU | 18.0 | 5.25 s | 11.1 | 0.62 | 21.1 s |
| Qwen3.5-4B Q4_K_M GGUF | iGPU, Vulkan | 21.8 | 0.34 s | 11.9 | 0.54 | 6.0 s |
| Qwen3.5-4B Q4_K_M GGUF | iGPU, ROCm | 18.1 | 0.51 s | 11.9 | 0.66 | 5.4 s |
| Qwen3.5-4B Q4_K_M GGUF | CPU | 14.3 | 0.70 s | 12.0 | 0.84 | 3.7 s |

- The NPU's speed doesn't change between AC and battery (15.8 tok/s both); the iGPU loses 12–25 % and the CPU 5 %.
- Per token the iGPU with Vulkan is still the most efficient on this machine. Use the NPU to keep the iGPU and
  CPU free (desktop, browser, compiles stay smooth while a model runs) and for always-there helpers
  (dictation, OCR, search), not because a single answer costs less energy.
- Whole-system draw is in the raw output (`sys`), but it moved with screen and background activity by more than
  the differences between backends, so the table uses package power.

## Idle cost

| State | Wakeups/s (flm + gateway) | Package power |
|---|---:|---:|
| model loaded, no requests | 0 + 0 | +0.4 W vs. unloaded; NPU runtime-suspended |
| nothing loaded | gateway exits after `exit_after`; only the systemd socket remains | baseline |

## Other NPU workloads

| Workload | Model | Speed |
|---|---|---|
| Speech to text | whisper-v3:turbo | 11 s clip → 2.9 s warm, ~5 s including model load |
| Embeddings | embed-gemma:300m | 2.4–3.7 chunks/s (60-line chunks), ~21 W package while indexing; ~1.5 s fixed cost per request, so batch. Full index of ~1000 files: 20 min, updates: seconds |
| OCR of a code screenshot | qwen3.5:4b / qwen3vl-it:4b | ~3 s prefill + ~15 tok/s |

## GPU backend (llama.cpp Vulkan): RX 9070 XT 16 GB, Ryzen 7 9800X3D, 32 GB RAM

`vram_reserve = 0.2` (`--fit-target 3260`), `threads = 6`, `-c 16384`, thinking off, a ~370-token answer.

| Model | Desktop VRAM in use | On the GPU | Generation | CPU while generating |
|---|---|---|---|---|
| Qwen3.5-9B Q4_K_M + mmproj (fast / vision) | 3.2 GB | all (6.5 GiB) | ~88 tok/s | ~1 core |
| Qwen3.6-35B-A3B UD-IQ3_XXS (quality) | 3.2 GB (light desktop) | 9.7 GiB, rest in RAM | 84.5 tok/s | 4.5 cores |
| same | + 4.8 GB (game running) | 5.0 GiB | 47.4 tok/s | 5.5 cores |
| same | + 7.0 GB (heavy game) | 2.8 GiB | 40.3 tok/s | 5.6 cores |

Loads take 3–9 s from the page cache. The 20% reserve stayed free in every case; the MoE expert weights that
don't fit live in RAM, which is why the 35B model stays fast with most of it off the GPU.
