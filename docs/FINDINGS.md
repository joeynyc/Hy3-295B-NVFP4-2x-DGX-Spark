# Findings — the experimental log

Everything below was measured on 2× DGX Spark (GB10, 121 GB unified each) over 200 GbE RoCE, serving a compressed-tensors NVFP4 quant of `tencent/Hy3-preview` (295B MoE / 21B active) on vLLM 0.22.1-dev (built from main, 2026-07-06) with Ray. Benchmark = 200 completion tokens, temperature 0, short prompt, second run after warmup unless noted.

## 1. TP=2 beats PP=2 for single-stream (+16%)

| Config | tok/s (3-run) |
|---|---|
| `-tp 2 -pp 1` | 13.9 / 14.7 / 14.6 → **14.4 avg** |
| `-tp 1 -pp 2` | 11.4 / 12.4 / 12.5 → 12.1 avg |

Same conditions, same day, back-to-back. Intuition says PP should win batch-1 (one hidden-state hop per token vs 2 allreduces per layer), but measurement says otherwise for this MoE on this fabric. **Keep TP=2.**

*(Absolute numbers in this table are from a memory-pressured system — see §6. The ratio held.)*

## 2. Jumbo frames: free, but not a decode win

Raising cluster-rail MTU 1500 → 9000 (RoCE path MTU 1024 → 4096) did **not** move batch-1 decode (13.9–14.7 before and after). Decode allreduces are tiny (~KB) and latency-bound, not bandwidth-bound. Kept anyway: helps prefill/batched traffic, costs nothing. Persist via netplan or it reverts on reboot.

## 3. fastsafetensors OOMs GB10 on ~80 GB/node models

`--load-format fastsafetensors` pins staging buffers that roughly double the weight footprint during load. 80 GB × 2 > 121 GB unified → kernel OOM that took down the worker's raylet AND sshd (node unreachable for minutes), twice. The default safetensors loader is slower (~10 min) but safe. **Never use fastsafetensors for big models on Spark.**

## 4. MTP speculative decoding: works after a patch, still a net loss cross-node

- Stock vLLM cannot load this checkpoint's MTP draft layer: `eh_proj` is a bare `nn.Linear`, quant emits packed params → `KeyError: …eh_proj.weight_global_scale`. Fixed by swapping to quant-aware `ReplicatedLinear` — upstreamed as [vllm#47792](https://github.com/vllm-project/vllm/pull/47792).
- With the fix: drafts flow, acceptance is decent (mean length 1.71, ~67% tokens accepted)…
- …and it's **slower**: 11.3 vs 14.5 tok/s. Every decode step now runs draft forward + verify forward, each paying cross-node allreduce latency. The interconnect tax exceeds the acceptance gain. Likely wins on single-node; loses on dual Spark.

## 5. Router `expert_bias` silently downcast (quality, not speed)

vLLM allocates Hy3's expert-selection bias in serving dtype (bf16); the checkpoint deliberately ships it fp32 (it's in the quant ignore-list). bf16's ~3 significant digits can flip near-tie expert choices → silent quality degradation. Reported as [vllm#47777](https://github.com/vllm-project/vllm/issues/47777); mitigated locally by `mods/fix-hy3-expert-bias-fp32.sh` (one line, fp32 allocation). DeepSeek models already pin the equivalent parameter to fp32 in vLLM.

## 6. Memory pressure halves throughput; a reboot restores it

After a day of loading/unloading 80 GB models, both nodes sat at 113–116/121 GB used with 1–3.5 GB swap. Every benchmark on that state: ~14.5 tok/s. After a (coincidental) reboot of both nodes: **27 tok/s single, 60 tok/s 4-way aggregate — everything exactly 2×.** Unified memory pressure taxes every decode step. If throughput sags and swap is non-zero, reboot.

## 7. `nvfp4` KV cache is impossible on Spark; TurboQuant is the real option

- `--kv-cache-dtype nvfp4` → `ValueError: --kv-cache-dtype nvfp4 requires sm100f`. Hard architecture gate (B200-class). GB10 is sm121. This is why community fp4-attention kernels (A4Q etc.) exist for Spark.
- `--kv-cache-dtype turboquant_k8v4` (8-bit K / 4-bit V, TURBOQUANT attention backend, forces FA2): **works on GB10, mainline vLLM, no patches.**

| KV dtype | Pool size | Single-stream | Quality spot-check |
|---|---|---|---|
| fp8_e4m3 | 310,656 tok | 26.5 tok/s | ✓ |
| turboquant_k8v4 | ~414K tok (+33%) | 25.6–26.0 tok/s | ✓ (exact 847×293, capitals, tool calls) |

Adopted `turboquant_k8v4` as default. 3-bit and 4-bit-both variants exist (`turboquant_3bit_nc`, `turboquant_4bit_nc`) — untested here.

## 8. Deep-context decode is bandwidth physics, not paging

191,508-token prompt (75% of the 256K window), fp8 KV era:

- Prefill: 385 s ≈ **500 tok/s**
- Decode at that depth: **9.4 tok/s** (vs 26.5 short-context)

Nothing pages — vLLM pre-allocates the whole KV pool at startup, and ~37 GB of KV stays resident. The slowdown is each token reading ~18 GB/node of KV through the same LPDDR5X that streams the weights. Raising `--max-model-len` costs zero memory (admission cap, not allocation); you only pay when a request actually fills the window.

## 9. Misc operational facts

- 256K native max (`--max-model-len 262144` + `VLLM_ALLOW_LONG_MAX_MODEL_LEN`); beyond native = RoPE extrapolation, unvalidated for this checkpoint, and the KV wouldn't fit at fp8 anyway.
- Concurrency scales well for an MoE: 1 stream = 27 tok/s; 4 streams = 15 each (60 total). `--max-num-seqs` is the knob.
- First post-restart benchmark is warmup (~4× slow). Always measure twice.
- Model quant used here: 0xSero NVFP4 of `tencent/Hy3-preview` — the *preview* checkpoint, not the final release. A final-release quant is the biggest available quality upgrade.
- Tool calling (`hy_v3` parser) verified: correct `tool_calls` finish, clean JSON args.
- Without a system prompt the model tends to answer in Chinese. Ship a system prompt.
