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

## Other NPU workloads

| Workload | Model | Speed |
|---|---|---|
| Speech to text | whisper-v3:turbo | 11 s clip → 2.9 s warm, ~5 s including model load |
| Embeddings | embed-gemma:300m | ~3.7 chunks/s (60-line chunks); ~1.5 s fixed cost per request, so batch |
| OCR of a code screenshot | qwen3.5:4b / qwen3vl-it:4b | ~3 s prefill + ~15 tok/s |
