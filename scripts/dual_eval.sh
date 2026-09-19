#!/usr/bin/env bash
# Print per-track eval averages for a dual-eval run.
set -u
RUN_ID="${1:?run id required}"
prime train metrics "$RUN_ID" --plain 2>/dev/null > /tmp/dual_eval_metrics.json
python3 - <<'EOF'
import json
data = json.load(open("/tmp/dual_eval_metrics.json"))
for row in data.get("metrics", []):
    step = row.get("step")
    evals = {
        k: round(v, 4)
        for k, v in row.items()
        if isinstance(v, (int, float)) and k.startswith("eval") and k.endswith("avg@2")
    }
    if evals:
        print(step, json.dumps(evals))
EOF
