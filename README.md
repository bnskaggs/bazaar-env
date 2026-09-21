# Bazaar Env

Status: **v1, v2, and v3 all trainer-verified** — three skills in one model
for $39.66 of hosted training. v1: a hosted GRPO run (Qwen3.5-9B on `micro`,
$10.88) raised the frozen-split eval from the passive baseline to the
honest-dealer anchor. v2: a curriculum run warm-started from the v1
checkpoint ($8.12) evals **above the vol-aware maker anchor** on
`maker_micro` with zero forgetting of the v1 skill — and plateaus fast,
which is stated plainly in `results/results.md`. v3: a mixed-batch curriculum
run ($7.80) clears the margin-taker anchor on `credit_micro` while improving
both prior skills. Scope: one run per training shape, one model, warm-started
after v1.

**v3 (credit).** Trainer-verified as a **three-skill curriculum** as of
09-21: one model that takes edges, makes markets, and sizes against a margin
loan. A mixed-batch curriculum run ($7.80) took `credit_micro` from 1.2899 to
1.3001 — **above the scripted margin-taker anchor** — while *improving* the
maker and taker evals rather than trading them away. A short credit-only
fine-tune ($3.03) also cleared the baseline, by less.

Two things about v3 are stated plainly rather than buried, because both are
more interesting than the headline:

- **The first credit run (09-19, $9.83) was published as a partial success and
  was not one.** A baseline eval added two days later showed it had finished
  *below* the checkpoint it warm-started from on all three tiers; the write-up
  had benchmarked the base model instead of the parent checkpoint. Full
  correction in `results/results.md`, and the rule it cost us is in
  `DESIGN.md`.
- **Most of v3's credit skill came from transfer, not credit training.** The
  v1→v2 curriculum checkpoint scores 1.2899 on `credit_micro` — a tier it has
  never seen — which is ~80% of the way from the base model to the anchor.
  The training runs closed the last tenth and crossed the anchor.

Bazaar is a short-horizon dealer-market environment for LLM agents. A model
trades one commodity against scripted quotes while the commodity's fair value
follows a seeded exogenous index. The terminal score marks inventory at that
index, so the agent cannot move the reward by trading against itself or pumping
its own book.

v1 (solo taker), v2 (maker + scripted adversary), and v3 (automatic margin
credit) are all implemented. Shorting and the multi-agent call auction remain
pre-registered in `DESIGN.md`.

## Why This Exists

Most public agent environments cover coding, browser use, or enterprise
workflow. Bazaar targets a thinner shelf: economic environments with deterministic
market mechanics, rewards that cannot be moved by the agent's own trades, and a
documented exploit pass.

The first build is deliberately small. A trainable LLM environment is useful only
if rollouts are cheap enough to sample in groups, so v1 keeps the task to 10
turns and a compact text protocol.

## Install

```powershell
uv sync --extra dev
```

The published wheel deliberately does **not** depend on `verifiers` or
`prime-sandboxes`. Hosted training images provide that stack. This package is a
pure plugin over the platform image; local development installs the dev extra.

## Run Tests

```powershell
uv run pytest
```

## Scripted Smoke Runs

```powershell
uv run python -m bazaar_env.core --tier micro --seed 0 --policy honest
uv run python -m bazaar_env.exploits --tier micro --seeds 20
uv run python -m bazaar_env.exploits --tier hard --seeds 20
```

## Model Probes

After tests pass, the next step is an Ollama band probe. The first target is
`qwen2.5:7b-instruct`, matching the Magic Sort training-thread protocol.

```powershell
vf-eval bazaar -m qwen2.5:7b-instruct -b http://localhost:11434/v1 -k OPENAI_API_KEY -n 5 -r 2 -a '{\"tier\":\"micro\"}' --disable-env-server
```

On Windows, avoid `--save-results` when the model name contains a colon.

## Implemented v1 Loop

Each turn:

1. Scripted counterparties post a bid/ask around the visible index.
2. The agent replies with a command: `buy Q`, `sell Q`, or `pass`. The parser
   extracts the last command found in the message (Magic Sort protocol), so
   surrounding reasoning text does not fail the turn.
3. The environment executes at the quote or returns `illegal: <reason>`.
4. The index advances.
5. State is re-marked and returned as compact JSON.

Terminal reward:

```text
net_worth_at_index / starting_net_worth
+ once-only first-valid-trade bonus
+ small format bonus
```

Starting net worth = starting cash + starting inventory marked at the first
index. `micro` starts the dealer with a small book so both edge directions
are tradable from turn 1.

The once-only trade bonus prevents the no-trade collapse case without becoming a
farm: once every rollout trades, GRPO's group baseline cancels it.

## Files

- `src/bazaar_env/core.py` — deterministic market engine and scripted policies.
- `src/bazaar_env/env.py` — `verifiers` adapter, datasets, replay scoring.
- `src/bazaar_env/exploits.py` — deterministic exploit pass.
- `DESIGN.md` — reward rationale, roadmap, and pre-registered farms.
- `results/results.md` — commands and a place for first eval numbers.

## V2: Maker + Adversary

Status: **v1 trainer-verified; v2 engine-tested**. v1 has one hosted GRPO run. v2 adds maker quoting and scripted adversary flow, with model preflights still pending.

On maker tiers (`maker_micro`, `maker_easy`) the agent can post a two-sided quote:

```text
quote BP BQ AP AQ
```

That means bid price/size and ask price/size. `pass` leaves the standing quote in place. After the quote is posted, the index steps forward and scripted flow trades against the standing quote:

- **Noise flow** fills attractive quotes.
- **Pickoff flow** sees the stepped index and takes stale quotes.
- **Bait depth** can make displayed size larger than executable size on juicy taker quotes.
- **Trigger hunting** can move displayed taker quotes just inside a threshold the agent has revealed in prior trades.

Scripted anchors confirm the mechanic: a tight quoter gets many fills and loses to pickoff; a wide quoter does nothing; a vol-aware quoter earns spread with near-zero pickoff on `maker_micro`.

## V3: Credit (automatic margin)

Status: **engine-tested; model preflight below.** On credit tiers
(`credit_micro`) the agent has an automatic margin loan: buy beyond your cash
and the loan is drawn for you (up to 1x equity); surplus cash repays it each
turn; interest accrues at 0.5%/turn on outstanding debt. If equity falls below
30% of debt: margin call, one turn to cure, then default (episode over,
terminal score 0). The terminal score is final equity over starting net worth.

`credit_micro` is deliberately capital-poor with rare, fat edges, and
taker-only (an outside review found a maker degeneracy; see DESIGN.md).
Measured anchors: sizing real edges to full buying power beats staying
unlevered (1.050 vs 1.043); max-size gambling loses to doing nothing (0.981
vs 0.998). `DESIGN.md` carries the leverage sweep, the four measured design
dead-ends that led to automatic margin, and the two outside-review findings.
