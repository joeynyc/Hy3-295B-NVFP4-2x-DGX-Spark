#!/bin/bash
# Fix hy_v3_mtp draft model failing to load NVFP4-quantized eh_proj
# (KeyError: model.layers.80.eh_proj.weight_global_scale).
# eh_proj is a bare nn.Linear; swap to quant-aware ReplicatedLinear.
set -e
python3 - << "PYEOF"
p = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/hy_v3_mtp.py"
src = open(p).read()
if "ReplicatedLinear" in src:
    print("mtp eh_proj fix: already applied")
    raise SystemExit(0)
src = src.replace(
    "from vllm.model_executor.layers.layernorm import RMSNorm",
    "from vllm.model_executor.layers.layernorm import RMSNorm\nfrom vllm.model_executor.layers.linear import ReplicatedLinear",
    1)
old = "        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)"
new = """        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.eh_proj",
        )"""
assert old in src
src = src.replace(old, new, 1)
old_fwd = """        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )"""
new_fwd = """        hidden_states, _ = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )"""
assert old_fwd in src
src = src.replace(old_fwd, new_fwd, 1)
open(p, "w").write(src)
print("mtp eh_proj fix: applied")
PYEOF
