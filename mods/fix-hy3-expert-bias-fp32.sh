#!/bin/bash
# Fix vllm-project/vllm#47777: HYV3 router expert_bias silently downcast to
# serving dtype; checkpoint ships it fp32 and routing decisions depend on it.
set -e
python3 - << "PYEOF"
p = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/hy_v3.py"
src = open(p).read()
old = "self.expert_bias = nn.Parameter(torch.empty(config.num_experts))"
new = "self.expert_bias = nn.Parameter(torch.empty(config.num_experts, dtype=torch.float32))"
if new in src:
    print("expert_bias fp32 fix: already applied")
elif old in src:
    open(p, "w").write(src.replace(old, new, 1))
    print("expert_bias fp32 fix: applied")
else:
    raise SystemExit("expert_bias pattern not found - vllm changed, check manually")
PYEOF
