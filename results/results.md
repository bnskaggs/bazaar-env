# Bazaar Results

Status: **trainer-verified (one LoRA run, one tier) as of 2026-09-18.**

## Training Run 1 — Qwen3.5-9B on micro (hosted LoRA GRPO)

Run `bazaar-env--qwen3.5-9b--boq0dl` (`boq0dlqme647ww0ryu7coud1`), env
`bnskaggs/bazaar-env@0.1.0`, config `configs/train-qwen35-9b-micro.toml`
(60 steps x 64 rollouts, groups of 8, max_tokens 400, thinking off).
Total cost **$10.88**. Launched and completed 2026-09-18; the env-server
stack passed health checks on the first attempt (pure-plugin packaging).

**Train reward (batch mean):** 1.249 -> ~1.29 plateau from step ~52
(peaks 1.297). The batch *minimum* is the early story: mins of 1.03-1.05
(blunder/format tail) vanish by step ~9, then the mean grinds upward —
first the bad tail is eliminated, then edge capture improves.

**Frozen-split eval (20 x 2, temp 0), monotonic rise:**

| Step | eval avg (reward) | implied terminal return |
|---|---|---|
| base | 1.2506 | ~1.001 (passive level) |
| 15 | 1.2521 | ~1.002 |
| 30 | 1.2751 | ~1.025 |
| 45 | 1.2836 | ~1.034 |
| 60 | **1.2861** | **~1.036** |

The trained policy lands essentially at the honest scripted anchor
(1.040 terminal return) — the model learned the compare-and-trade
behavior it could not execute zero-shot (preflight: 1.012). No late-run
eval dip this time; step 60 is the keeper.

**Design claims this validates:** dense same-seed GRPO groups carried
the gradient despite a 0/8 "solve rate" at the (miscalibrated) 1.05x
threshold — the pre-registered "no bootstrap" risk did not materialize.
Occasional train rollouts beat the greedy anchor (batch max ~1.33 ≈
1.08 return), consistent with sizing/timing improvements beyond greedy
edge-taking; the temp-0 eval does not exceed the anchor.

Scope stated plainly: one run, one model, one tier. "Trainer-verified"
means the train split, shaped rewards, and group variance have executed
and produced a rising curve — nothing more.

## Deterministic Exploit Pass

Run:

```powershell
uv run python -m bazaar_env.exploits --tier micro --seeds 20
uv run python -m bazaar_env.exploits --tier hard --seeds 20
```

Record:

- Passive baseline terminal return.
- Honest dealer terminal return.
- Coin-flip gambler terminal return.
- Loop-spammer stop count.

### 2026-09-18 scripted smoke (after review fixes)

Review fixes applied first: lenient last-match action parsing (Magic Sort
protocol) and the no-progress stop narrowed to repeated identical
illegal/unparseable messages only — repeated legal passes are progress, so
buy-and-hold is never stopped early.

`micro`, 20 seeds:

```json
{
  "gambler": {"mean_terminal_return": 1.0200495, "mean_num_trades": 8.15, "no_progress_stops": 0, "wins_vs_passive": 0.75},
  "honest": {"mean_terminal_return": 1.0526935, "mean_num_trades": 7.3, "no_progress_stops": 0, "wins_vs_passive": 1.0},
  "loop_spammer": {"mean_terminal_return": 1.0, "mean_num_trades": 0, "no_progress_stops": 20},
  "passive": {"mean_terminal_return": 1.0, "mean_num_trades": 0, "no_progress_stops": 0}
}
```

`hard`, 20 seeds:

```json
{
  "gambler": {"mean_terminal_return": 0.956938, "mean_num_trades": 8.35, "no_progress_stops": 0, "wins_vs_passive": 0.15},
  "honest": {"mean_terminal_return": 0.9976635, "mean_num_trades": 1.0, "no_progress_stops": 0, "wins_vs_passive": 0.3},
  "loop_spammer": {"mean_terminal_return": 1.0, "mean_num_trades": 0, "no_progress_stops": 20},
  "passive": {"mean_terminal_return": 1.0, "mean_num_trades": 0, "no_progress_stops": 20}
}
```

Read: `micro` has the intended cheap edge — honest beats passive on **20/20
seeds** (+5.3% avg) and beats the coin-flip gambler (+3.3pp), so
variance-seeking does not pay at the easy end. At `hard` both active policies
lose to passive on average (thin edges + vol) — expected; it serves as the
high-end exploit check. Only the loop-spammer triggers the no-progress stop.

## Ollama Band Probe

Target model:

```text
qwen2.5:7b-instruct
```

First run:

```powershell
vf-eval bazaar -m qwen2.5:7b-instruct -b http://localhost:11434/v1 -k OPENAI_API_KEY -n 5 -r 2 -a '{\"tier\":\"micro\"}' --disable-env-server
```

Windows note: do not use `--save-results` with colon-containing model names.

Band hypothesis:

- `micro` should produce enough trades and return variance for training.
- Success threshold: terminal return >= 1.02 (half the honest-anchor
  margin over passive; re-based 2026-09-18 after the original 1.05x was
  found to sit above the honest anchor itself — see the preflight
  ladder below).
- Edge-width is the first dial to tune if the band misses.

### 2026-09-18 first band probe — qwen2.5:7b-instruct, micro, 5x2

Plumbing smoke (n=2) plus full probe (n=5, r=2), local Ollama, ~10s/rollout:

```text
terminal_return: avg 1.007, std 0.014, range [0.983, 1.025]
solves at >=1.05x threshold: 0/10   (honest scripted anchor: 1.053)
num_trades: avg 2.8   first_trade_turn: 1.0 across all rollouts
illegal_count: avg 1.2/episode   format_reward: 1.0   no_progress_stops: 0
```

Read:

- **Protocol is a non-issue** (the seg-4 confound is absent): zero format
  failures, first trade always turn 1, no loops. The lenient parser and
  named illegal reasons did their job.
- **The model trades actively but captures no edge**: it hits quotes with
  negative `edge_vs_index` (observed buying at 101.1 vs index 100.0)
  instead of passing. Failure is judgment, not protocol — the intended
  trainable skill.
- **Band verdict: below band at the 1.05x threshold (0/10)**, but the
  reward is dense (std 0.014, real behavioral spread 0.983-1.025), so the
  gradient is thin rather than dead — unlike a sparse-solve env at 0%.
- **Structural finding for calibration:** sell edges (bid > index) are
  unusable while inventory is zero, so the effective edge frequency in the
  early episode is roughly half the nominal `edge_probability`. Options
  considered: (a) start the agent with inventory so both edge
  directions are live from turn 1, (b) bias early edges toward the buy
  side, (c) raise `edge_width`/`edge_probability`, (d) lower the
  threshold. (a) was adopted as the structurally correct dealer setup.

### 2026-09-18 calibration sweep — three dials, one conclusion

All free local probes, qwen2.5:7b-instruct, 5x2 each:

| Config | terminal_return | Notes |
|---|---|---|
| micro (empty book, baseline) | 1.007 ± 0.014 | trades actively, negative-edge fills |
| micro + starting_inventory=5 (now default) | 1.007 ± 0.009 | accumulates inventory on noise |
| micro + edge_width 8, edge_prob 0.95 | 1.012 ± 0.009 | grosser edges barely help |
| micro + `strategy_hint=true` (atom stated in rules) | 1.005 ± 0.008 | explicit instruction doesn't transfer to execution |

**Conclusion: the local quantized 7B does not execute the compare-and-trade
atom under any world configuration — the dial isn't the constraint, the
model is.** This mirrors Magic Sort exactly: local quantized 7B solved 0/8
on `trivial` there while hosted bf16 Qwen3.5-9B trained cleanly (0.60 →
1.80). The local 7B is a floor detector, and micro is below its floor.

The reward surface itself is proven live: honest scripted anchor 1.040,
dense rewards, real behavioral spread, protocol clean across all 40 LLM
rollouts (zero loops, ~1-2 illegal/episode, format ≈ 1.0).

**Recommended next band check: hosted preflight on Qwen3.5-9B (bf16) —
costs cents, same protocol as the Magic Sort preflights.** Changes kept
from the sweep: `starting_inventory=5` on micro (structurally correct
regardless of band) and `strategy_hint` env arg (documented instruction-
explicitness knob, default false).

### 2026-09-18 hosted preflights + sanity ladder (micro)

Prime Inference bf16 (4x2, max-tokens 400, thinking off) plus a gpt-5-nano
sanity probe (4x2, OpenAI). Total probe spend across the whole day: well
under $1.

| Policy / model | terminal_return | Notes |
|---|---|---|
| passive scripted | 1.000 | baseline |
| qwen2.5:7b-instruct (local, quantized) | 1.007 ± 0.014 | noise trading, hoards inventory |
| Qwen/Qwen3.5-9B (hosted bf16) | 1.012 ± 0.012 | flattens book by end — dealer instinct, no edge capture; wider edges don't help (1.006) |
| gpt-5-nano (reasoner) | **1.028 ± 0.019** (best 1.044) | 9 trades/ep, 0.25 illegal/ep — real edge capture |
| honest scripted dealer | 1.040 | greedy anchor |

**Findings:**

1. **The environment is sound.** The ladder orders exactly by model
   capability, a reasoning model approaches the greedy anchor by pure
   inference, and protocol failures are ~zero everywhere. The task
   measures what it claims to measure.
2. **The 1.05x solve threshold was miscalibrated by construction** — it
   sits *above* the honest greedy anchor (1.040). No policy that trades
   only visible edges can clear it on average. Threshold needs re-basing
   against the honest anchor (e.g., solve = beat passive by half the
   honest margin, ~1.02, per-seed).
3. **The GRPO case for training 9B despite "0/8":** rewards are dense and
   GRPO groups share a seed — the index path and quotes are identical
   within a group, so world noise cancels and within-group reward
   differences are pure policy signal. Headroom to the honest anchor is
   +0.028 from the 9B's 1.012, and nano proves the gap is closable by
   inference alone. Stated risk: the 9B currently samples no
   edge-directed behavior; if none appears in rollouts, the curve
   plateaus at noise — which would itself be a publishable finding about
   dense-reward trainability claims.

## Status Upgrade Rule

Do not call the environment trainer-verified until a hosted training run consumes
the train split and shows a rising reward curve. Passing tests and `vf-eval`
only prove the eval path.

## V2 Scripted Anchors (maker + adversary)

Status: **engine-tested; model preflight pending**.

Run:

```powershell
uv run --extra dev python -m bazaar_env.exploits --tier maker_micro --seeds 20
uv run --extra dev python -m bazaar_env.exploits --tier maker_easy --seeds 20
```

### 2026-09-18 anchors

`maker_micro`, 20 seeds:

```json
{
  "passive": {"terminal_return": 0.999, "fills": 0, "pickoff_losses": 0.0},
  "wide_quoter": {"terminal_return": 0.999, "fills": 0, "pickoff_losses": 0.0},
  "tight_quoter": {"terminal_return": 0.990, "fills": 57.15, "pickoff_losses": 14.606},
  "vol_aware_quoter": {"terminal_return": 1.020, "fills": 18.0, "pickoff_losses": 0.004},
  "threshold_taker": {"terminal_return": 1.026, "wins_vs_passive": 0.95}
}
```

`maker_easy`, 20 seeds:

```json
{
  "passive": {"terminal_return": 0.998, "fills": 0, "pickoff_losses": 0.0},
  "wide_quoter": {"terminal_return": 0.998, "fills": 0, "pickoff_losses": 0.0},
  "tight_quoter": {"terminal_return": 0.964, "fills": 66.75, "pickoff_losses": 41.635},
  "vol_aware_quoter": {"terminal_return": 1.002, "fills": 4.85, "pickoff_losses": 0.443},
  "threshold_taker": {"terminal_return": 1.007, "wins_vs_passive": 0.90}
}
```

Read: the mechanics are live. Tight quoting overtrades and gets picked off;
wide quoting earns nothing; the vol-aware maker earns spread with low pickoff on
`maker_micro`. `maker_easy` is the hard-end exploit check: tight quoting gets
punished heavily and vol-aware becomes safe but sparse.

## V2 Preflight Protocol

First probes:

```powershell
uv run --extra dev vf-eval bazaar -m gpt-5-nano -b https://api.openai.com/v1 -k OPENAI_API_KEY -n 4 -r 2 -a '{"tier":"maker_micro"}' --disable-env-server
uv run --extra dev vf-eval bazaar -m Qwen/Qwen3.5-9B -b https://api.pinference.ai/api/v1 -k PRIME_API_KEY -n 4 -r 2 --max-tokens 400 -S '{"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' -a '{"tier":"maker_micro"}' --disable-env-server
```

Band read should compare models to the `vol_aware_quoter` anchor (terminal return
~1.020 on `maker_micro`), not to v1's taker anchor.

### 2026-09-18 v2 model preflights (maker_micro)

4 examples x 2 rollouts. Nano via OpenAI; Qwen3.5-9B via Prime Inference
(bf16, max_tokens 400, thinking off).

| Model | terminal_return | fills | realized_spread | pickoff_losses | Read |
|---|---:|---:|---:|---:|---|
| gpt-5-nano | 1.012 +/- 0.003 | 1.5 | 1.03 | 0.07 | Mostly falls back to taker behavior; one rollout posts/fills. |
| Qwen/Qwen3.5-9B | 1.004 +/- 0.008 | 13.8 | 5.52 | 5.78 | Makes markets, but gives back the spread to pickoff. |
| vol-aware scripted anchor | 1.020 | 18.0 | 31.25 | ~0 | Honest-maker target. |

Verdict: **v2 is engine-tested, not trainable yet.** The mechanics are live and
measurable (fills, spread, pickoff), but neither model clears the vol-aware
anchor. This is a good eval surface: the 9B learns/executes "post quotes" but
not "quote wide enough for volatility." A training run is **not** launched from
this band read. Next design dial is either easier maker_micro (lower vol / wider
honest anchor gap) or a staged curriculum from taker-trained weights to maker.

### 2026-09-18 v2 review fixes (0.2.1) -- supersedes the anchor numbers above

A review pass found five bugs in 0.2.0; all fixed and regression-tested
(39 tests total):

1. Taker trades did not record `edge_vs_index`, so trigger-hunting (con 3)
   could never fire in live play. Fixed; an end-to-end test now proves a real
   profitable take arms the hunt.
2. Posted quote size did not bound per-turn exposure (noise ignored size;
   pickoff looped `intensity` times at full size). Fixed: posted size is a hard
   per-turn per-side cap across all flow. Pickoff now arrives with probability
   `1 - 0.5^pickoff_intensity` and takes the remaining stale size once.
3. Maker fills recorded `turn+2` in the trade log the model reads. Fixed.
4. Maker-only state keys leaked into v1 tiers, silently changing the v1 prompt
   format the trained run was evaluated on. Fixed: v1 state JSON is
   byte-identical to the training run's format again.
5. `avg_quote_width` measured fill-edge dispersion, not width. Fixed: it now
   averages `ask - bid` across posted quote commands.

Post-fix anchors (20 seeds): `maker_micro` tight 0.996 (losses 5.4, bounded),
vol-aware 1.020 (unchanged), wide/passive ~0.999; `maker_easy` tight 0.983
(losses 13.7), vol-aware 1.002. The 0.2.0 preflight numbers above were measured
on the buggy engine (uncapped exposure, dead con 3); the 9B preflight is re-run
below on 0.2.1.

### 2026-09-18 9B re-preflight on 0.2.1 (maker_micro, fixed engine)

| Model | terminal_return | fills | realized_spread | pickoff_losses | avg_quote_width |
|---|---:|---:|---:|---:|---:|
| Qwen/Qwen3.5-9B | 1.008 +/- 0.005 | 15.5 | 6.79 | 2.89 | ~1.0 |
| vol-aware scripted anchor | 1.020 | 17.9 | 31.18 | ~0 | ~3.8 |

Read: better than the buggy-engine read (1.004) and the model now makes real
markets (fills, spread capture), but it quotes about 1.0 wide where the
vol-aware anchor quotes ~3.8 wide -- it earns flow and gives too much of it
back to pickoff. Same verdict: **engine-tested, not trainable yet at this
tier.** The gap (quote wider / manage staleness) is exactly the intended
skill, so the candidate dials are: gentler maker_micro (lower vol or lower
pickoff intensity), or curriculum from the v1-trained checkpoint.

## V2 Curriculum Run (0.2.1) -- maker_micro from the v1 checkpoint

Run `bazaar-env--qwen3.5-9b--e7naqk` (`e7naqkjeix1hmqf9eibh264t`), warm-started
from the v1 step-55 checkpoint, 40 new steps on `maker_micro`, dual eval.
Total cost **$8.12**.

| Step | maker_micro eval | micro eval (forgetting) |
|---|---:|---:|
| 60 | 1.2752 | 1.2855 |
| 70 | 1.2791 | 1.2874 |
| 80 | 1.2790 | 1.2874 |
| 90 | 1.2792 | 1.2877 |

Reference points (reward scale = terminal + 0.25): vol-aware scripted anchor
~1.270; base-9B zero-shot preflight ~1.258.

**Findings:**

1. **Curriculum transfer works.** The v1-trained checkpoint opens the maker
   game at the anchor level (batch means 1.274 at step 56) and evals **above
   the vol-aware anchor** (~1.029 terminal vs 1.020) where the base model
   could not reach it zero-shot (1.008).
2. **Zero forgetting.** The micro (v1 taker) eval holds at 1.2877, matching
   the v1 run's final 1.2861.
3. **Fast plateau, stated plainly.** The maker eval improved for ~10 steps
   then moved +0.0001 over the final 20. Most of the lift is the transferred
   prior; the residual trainable headroom on maker_micro with a mild
   adversary is small. Pushing further needs a harder tier (higher vol /
   pickoff intensity, maker_easy) or a tighter anchor to chase.

**Status upgrade: v2 is trainer-verified (one curriculum run, one tier)** --
the train split, maker rewards, and group variance have executed and produced
an above-anchor policy with no catastrophic forgetting. Scope: one run, one
model, one tier, warm-started.

## V3 Credit Anchors (0.3.0, credit_micro)

Status: **engine-tested; preflights below.**

Run:

```powershell
uv run --extra dev python -m bazaar_env.exploits --tier credit_micro --seeds 20
uv run --extra dev python -m bazaar_env.exploits --lottery --seeds 20
uv run --extra dev python scripts/leverage_sweep.py
```

### 2026-09-19 anchors (20 seeds)

```json
{
  "margin_taker": {"terminal_return": 1.0371, "interest": 6.24, "defaults": 0},
  "unlevered_taker": {"terminal_return": 1.0292, "interest": 0.0},
  "honest_taker": {"terminal_return": 1.0315, "interest": 0.0},
  "vol_aware_quoter": {"terminal_return": 1.047, "interest": 3.35},
  "wide_quoter": {"terminal_return": 1.0538, "interest": 0.35},
  "margin_gambler": {"terminal_return": 0.9734, "interest": 8.49, "defaults": 0},
  "passive": {"terminal_return": 0.9977}
}
```

Reads: leverage amplifies skill (margin taker +0.8pp over unlevered); the
limited-liability lottery loses at shipped settings and at high vol (0.984 vs
0.988 passive); the leverage sweep in DESIGN.md shows the lottery leaking at
2-3x before defaults claw it back at 4x, which is why max_leverage ships at
1.0. Note the maker anchors on this tier: rare fat edges make wide quoting
strong (1.0538) - the tier's best play mixes making and levered taking.

Honesty note: zero defaults occur in anchor play at 1x leverage; the
covenant/default machinery is unit-tested and sweep-exercised (first default
observed at 4x), and exists as the boundary that makes the leverage dial safe
to raise.

### 2026-09-19 9B preflight on credit_micro (zero-shot)

```text
terminal_return: avg 1.009, std 0.041 (richest spread of any tier; v1 micro was 0.014)
best rollout: 1.044 (above the margin-taker anchor)
margin use: real and unprompted - max_debt up to 670, interest paid, zero defaults
anchors: passive 0.998 / margin_taker 1.037 / wide_quoter 1.054
```

Read: above passive, well below the anchors, with high within-task variance and
behaviors partially present. This is the most trainable-shaped band read Bazaar
has produced. Recommended training shape if funded: warm-start from the v2
curriculum checkpoint (three-stage curriculum: taker -> maker -> credit).

### 2026-09-19 outside review (0.3.1) -- supersedes the 0.3.0 credit numbers

Two review findings fixed: credit_micro is now TAKER-ONLY (a static wide quote
earned 1.054 risk-free and ignored credit entirely -- the noise-reservation /
edge-width coupling; documented in DESIGN.md) and the terminal score is floored
at 0 with or without default (surviving negative equity no longer scores below
defaulting). 55 tests.

Post-fix anchors (20 seeds): margin_taker 1.0495 / unlevered_taker 1.0428 /
honest_taker 1.0446 / margin_gambler 0.9811 / passive 0.9977.

9B re-preflight (taker-only tier): terminal 1.003 +/- 0.017, margin used
unprompted in some rollouts, zero defaults, zero illegal loops. The prior
1.009 read was inflated by maker fills on the degenerate tier. Anchor gap to
margin_taker (~4.6pp) is the largest of any Bazaar tier -- trainable-shaped;
recommended run remains the three-stage curriculum warm-start.

## V3 Curriculum Run -- credit_micro from the v2 maker checkpoint

Run `bazaar-env--qwen3.5-9b--dqp67h` (`dqp67hcmadgq8t2mhov0247t`), warm-started
from the v2 step-90 checkpoint (`ix2fgbi0csvgvkazkedvmkva`), trained on
`credit_micro` for 40 new steps (90 -> 130), triple eval. Total cost **$9.83**.

| Step | credit_micro eval | maker_micro eval | micro eval |
|---|---:|---:|---:|
| 105 | 1.2655 | 1.2629 | 1.2660 |
| 120 | **1.2758** | 1.2633 | 1.2717 |

Reference points (reward scale):

- Base 9B zero-shot on credit_micro: ~1.253 (terminal 1.003).
- margin_taker anchor: ~1.2995 reward (terminal 1.0495 + 0.25 bonuses).
- v2 maker run final: maker_micro 1.2792, micro 1.2877.

**Findings:**

1. **Credit learning exists but is incomplete.** credit_micro eval improves by
   +0.0103 reward (~+1pp terminal return) over 15 steps and clears the zero-shot
   baseline, but remains below the margin_taker anchor.
2. **Forgetting is real.** maker_micro collapses from 1.2792 to ~1.263 and micro
   from 1.2877 to 1.272. The three-stage curriculum did not preserve prior
   skills under single-tier credit training.
3. **Train reward degrades after the early steps.** Batch means open high
   (1.33 at step 91, 1.34 at step 98) but slide toward ~1.24-1.26 by the end.
   The late checkpoint is not an obvious keeper; step 120 has the only credit
   eval improvement but still forgets.

**Verdict:** v3 is **trained, but not cleanly trainer-verified as a stable
three-stage curriculum.** The credit tier is a live learning signal, but the
single-tier run trades off prior skills. Next attempt should use a mixed
curriculum (credit_micro + maker_micro + micro in the training batch, not only
in eval) or shorter credit fine-tuning from the v2 checkpoint.
