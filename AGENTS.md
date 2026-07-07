# AGENTS.md — operational runbook for AI agents

This file is written for AI agents (Claude Code, Codex, Pi, etc.) operating this cluster or answering questions about it. Facts here are measured, not aspirational. Prefer commands listed here over improvising.

## System identity

- **Model**: Tencent Hy3 (Hunyuan 3), 295B MoE / 21B active, NVFP4 weight quant (compressed-tensors `nvfp4-pack-quantized`), served from a local path on both nodes (default `/models/Hy3-NVFP4` inside containers, `~/models/Hy3-NVFP4` on hosts).
- **Topology**: 2× NVIDIA DGX Spark (GB10, 121 GB unified memory each). Tensor parallel = 2 via Ray. Head node runs the API server and TP rank 0; worker runs TP rank 1. Cluster link: 200 GbE RoCE, MTU 9000, cluster IPs on `enp1s0f1np1`.
- **API**: OpenAI-compatible, head node port 8000. Model id: `/models/Hy3-NVFP4`.
- **Containers**: name `vllm_node` on both hosts, host networking, privileged, started with `--rm` (they do NOT survive reboot — that is expected, not a fault).

## Expected numbers (deviation = investigate)

| Measurement | Healthy value |
|---|---|
| Single-stream decode, short prompt | 25–27 tok/s |
| Same, FIRST run after restart | 6–20 tok/s (warmup — always bench twice) |
| 4 concurrent streams | ~15 tok/s each, ~60 aggregate |
| Decode at ~190K context depth | ~9–10 tok/s (normal physics, not a fault) |
| Prefill | ~500 tok/s |
| Cold model load to API-up | 10–15 min |
| max_model_len from `/v1/models` | 262144 |
| Host RAM in use while serving | ~110–116 GB of 121 GB per node |

If sustained single-stream drops well below ~20 tok/s at short context AND swap usage is multi-GB on either node: the fix is a reboot of both nodes (measured recovery: 14 → 27 tok/s). Restart serving afterward with `start.sh`.

## Health checks (safe, read-only)

```bash
# Is the API up and what context does it report?
curl -s -m 5 http://<HEAD_IP>:8000/v1/models | jq '.data[0] | {id, max_model_len}'

# Full status: API, Ray membership, memory+swap on both nodes, link MTU
./scripts/hy3ctl status

# Deterministic 200-token benchmark (temperature 0). Run twice; trust the second.
./scripts/hy3ctl bench

# Ray cluster membership (expect 2 active nodes, 2/2 GPUs in use while serving)
docker exec vllm_node ray status
```

## Start / stop

```bash
./start.sh    # full bring-up: containers both nodes → Ray → mods → vLLM → wait → bench
./stop.sh     # tear down containers on both nodes
```

Rules an agent MUST follow:

1. **Never start a new vLLM launch while the previous model is still resident.** After killing vLLM, poll `free -g` on BOTH nodes until used memory is below ~20 GB before relaunching. Violating this OOMs the worker (~80 GB old + ~80 GB new > 121 GB) and can take down its raylet and sshd.
2. **Never use `--load-format fastsafetensors`** for this model on GB10. Pinned staging doubles the weight footprint during load → hard OOM. (Two confirmed incidents.)
3. **Do not enable MTP speculative decoding expecting speed.** It loads (with `mods/fix-hy3-mtp-nvfp4-eh-proj.sh`) but measured 20% SLOWER across two nodes (11.3 vs 14.5 tok/s, acceptance 1.7) — the draft pass adds a second cross-node allreduce per step.
4. **Do not set `--kv-cache-dtype nvfp4`.** Hard-gated upstream: `ValueError: --kv-cache-dtype nvfp4 requires sm100f`. GB10 is sm121. The working sub-8-bit option on this hardware is `turboquant_k8v4` (validated at full speed on this model).
5. A restart takes 10–15 minutes of downtime. Do not restart casually; check with the human if the server is likely in active use.

## Failure signatures → resolutions

| Signature | Meaning | Resolution |
|---|---|---|
| `The actor died because its node has died … missing too many heartbeats` | Worker OOM or raylet death | Verify worker over cluster link; when memory is free, rejoin Ray (below), then relaunch vLLM |
| API 200 locally on head but connection-refused/reset from LAN | Head's LAN NIC wedged (link up, zero packets) — has happened post-reboot | Bounce the NIC. Via privileged container if no sudo: toggle IFF_UP with a python ioctl inside `vllm_node` (host netns). Cluster link keeps working throughout |
| SSH to a node fails with `Couldn't open /dev/null: Permission denied` | Known head-node bug: reboot restores `/dev/null` with 660 perms | `sudo sh -c 'rm -f /dev/null && mknod -m 666 /dev/null c 1 3'` |
| Both nodes rebooted, nothing serving, containers gone | Power event; containers are `--rm` | `/dev/null` fix if needed, then `./start.sh` |
| First bench after restart ~6 tok/s | Warmup | Bench again |
| `KeyError: model.layers.80.eh_proj.weight_global_scale` at load | MTP requested without the eh_proj mod | Apply `mods/fix-hy3-mtp-nvfp4-eh-proj.sh` in BOTH containers, or remove `--speculative-config` |
| vLLM/Ray killed at high memory with no kernel OOM log | `earlyoom` (stock Spark config prefers killing inference) | `sudo systemctl stop earlyoom` or move vllm/ray to its `--avoid` list |

**Rejoin a dropped Ray worker** (run on worker host):

```bash
docker exec vllm_node ray stop --force
docker exec -d vllm_node bash -c "ray start --block --object-store-memory 1073741824 \
  --num-cpus 2 --disable-usage-stats --address=<HEAD_IP>:29501 \
  --node-ip-address <WORKER_IP> >> /proc/1/fd/1 2>&1"
```

A pending vLLM placement group will grab the GPUs automatically once the worker rejoins.

## Request conventions

- Set a **system prompt** — without one the model often answers in Chinese.
- Thinking: `"chat_template_kwargs": {"enable_thinking": true|false}`. Off = fast terse replies; on = reasoning traces (streamed as `reasoning_content` deltas).
- Tool calling works natively (`hy_v3` parser); expect `finish_reason: "tool_calls"`.
- Prefix caching is on: keep system prompts byte-stable across turns for fast TTFT.
- Give generous `max_tokens` (≥1024): this preview checkpoint rambles into small caps.

## Benchmarks an agent can run

```bash
./scripts/hy3ctl bench                 # single-stream, 200 tok, temperature 0
# 4-way concurrency (aggregate should be ~55-60 tok/s):
seq 1 4 | xargs -P4 -I{} curl -s http://<HEAD_IP>:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/Hy3-NVFP4","prompt":"Write a story {}:","max_tokens":150}' -o /dev/null
python3 scripts/bench-longctx.py       # deep-context prefill + decode measurement
```

## Provenance

Configuration decisions and every number above: see [docs/FINDINGS.md](docs/FINDINGS.md). Upstream work: [vllm#47792](https://github.com/vllm-project/vllm/pull/47792) (our PR), [vllm#47777](https://github.com/vllm-project/vllm/issues/47777) (bug we mitigate via `mods/`).
