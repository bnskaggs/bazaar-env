"""Preflight a `tier_mix` training dataset before paying for a run.

Checks the three things that can silently ruin a mixed run: the weighting is
what the config author intended, every task is distinct (repeated tiers must
draw from separate seed blocks), and credit rules appear only on credit rows.
"""

from collections import Counter

from bazaar_env.env import load_environment

env = load_environment(
    tier_mix="credit_micro,credit_micro,maker_micro,micro",
    num_train_examples=320,
    num_eval_examples=20,
)

train_info = env.dataset["info"]
eval_info = env.eval_dataset["info"]

print("train rows:", len(env.dataset), Counter(r["tier"] for r in train_info))
print("eval rows:", len(env.eval_dataset), Counter(r["tier"] for r in eval_info))
print("train seed range:", min(r["seed"] for r in train_info), "-", max(r["seed"] for r in train_info))
print("unique train tasks:", len({(r["tier"], r["seed"]) for r in train_info}))
print("first 8 tiers:", [r["tier"] for r in train_info[:8]])

for idx in range(4):
    system = env.dataset[idx]["prompt"][0]["content"]
    print(f"row {idx} tier={train_info[idx]['tier']} credit_rules={'MARGIN CALL' in system}")
