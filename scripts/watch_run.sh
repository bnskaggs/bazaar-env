#!/usr/bin/env bash
# Monitoring helper for hosted training runs (run via WSL).
# Usage: watch_run.sh <run_id>
set -u
RUN_ID="${1:?run id required}"
prime train metrics "$RUN_ID" --plain 2>/dev/null > /tmp/watch_run_metrics.json
python3 - <<'EOF'
import json
data = json.load(open("/tmp/watch_run_metrics.json"))
for row in data.get("metrics", []):
    step = row.get("step")
    out = {}
    for k, v in row.items():
        if not isinstance(v, (int, float)):
            continue
        kl = k.lower()
        if kl.endswith(("rewards/reward/mean", "rewards/reward/std", "rewards/reward/min", "rewards/reward/max")):
            out[k.split("/")[-1]] = round(v, 4)
        elif "/all/avg@" in kl and kl.startswith("eval"):
            out["eval_avg"] = round(v, 4)
    if out:
        print(f"step {step}: {json.dumps(out)}")
EOF
