#!/usr/bin/env bash
# Status + per-track eval for the two v3 follow-up runs (4a short, 4b mixed).
# Both branch off the same v2 step-90 checkpoint, so the tracks are directly
# comparable and belong side by side.
set -u

SHORT="${SHORT_RUN:-xrxif3tzpzb2f1e93o6ph7sl}"
MIXED="${MIXED_RUN:-nfa92qg1mguwnv5i0e7w2fah}"

for pair in "4a-short:$SHORT" "4b-mixed:$MIXED"; do
  label="${pair%%:*}"
  run="${pair##*:}"
  echo "=== $label ($run) ==="
  prime train get "$run" --plain 2>/dev/null | grep -iE 'status|step' | head -4
  prime train components "$run" --plain 2>/dev/null | tail -8
  prime train metrics "$run" --plain 2>/dev/null > "/tmp/metrics_$label.json"
  python3 - "$label" <<'EOF'
import json
import sys

label = sys.argv[1]
try:
    data = json.load(open(f"/tmp/metrics_{label}.json"))
except Exception as exc:  # noqa: BLE001
    print(f"  no metrics yet ({exc})")
    raise SystemExit

rows = data.get("metrics", [])
if not rows:
    print("  no metrics yet")
    raise SystemExit

for row in rows:
    step = row.get("step")
    evals = {
        key.replace("eval/", "").replace("/avg@2", ""): round(value, 4)
        for key, value in row.items()
        if isinstance(value, (int, float)) and key.startswith("eval") and key.endswith("avg@2")
    }
    train = row.get("train/reward/mean") or row.get("reward/mean")
    parts = []
    if train is not None:
        parts.append(f"train {round(train, 4)}")
    if evals:
        parts.append(json.dumps(evals))
    if parts:
        print(f"  step {step}: " + "  ".join(parts))
EOF
  echo
done
