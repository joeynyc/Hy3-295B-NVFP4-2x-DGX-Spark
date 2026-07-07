#!/bin/bash
# PROFILE: Tencent Hy3-NVFP4 — TP=2 single-stream, 256K context, TurboQuant k8v4 KV
# Run inside the vLLM container on the Ray head node.
# Changes vs comfort-tp2:
#   - NOTE: fastsafetensors REMOVED — its pinned staging doubles ~80GB weights during load and OOMs the 121GB unified memory on GB10
#   - max-num-seqs 4 (agent/tool workloads benefit; KV budget allows it at 64K)
#   - optional NCCL dual-rail spread (uncomment to A/B test)

export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_SKIP_INIT_MEMORY_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Experiment: spread allreduce across QPs / both RoCE rails.
# Benchmark with and without — helps some workloads, hurts others.
# export NCCL_IB_QPS_PER_CONNECTION=2
# export NCCL_IB_SPLIT_DATA_ON_QPS=1

vllm serve /models/Hy3-NVFP4 \
    --max-model-len 262144 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.88 \
    --kv-cache-dtype turboquant_k8v4 \
    --block-size 64 \
    --port 8000 \
    --host 0.0.0.0 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hy_v3 \
    --reasoning-parser hy_v3 \
    --trust-remote-code \
    -tp 2 \
    -pp 1 \
    --distributed-executor-backend ray
