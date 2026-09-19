# Bazaar Design

Status: **v1 implemented, eval-pending**.

This document records the design rulings before model rollouts. It is meant to
make the environment auditable: what the reward measures, what it deliberately
does not measure, and which farms are pre-registered.

## Design Rulings

- **Fiction:** solo dealer in one commodity. Fundamental value follows a seeded
  exogenous index. Scripted counterparties quote around it with noise and bias.
  The vocabulary is neutral, not crypto-specific.
- **State:** compact JSON containing cash, inventory, current quotes,
  turn/turns_remaining, recent trades, visible index value, and mark-to-index
  P&L.
- **Actions:** v1 is taker-only: `buy Q`, `sell Q`, `pass`. No shorting until
  the margin/credit version.
- **Clearing:** v1-v3 are quote-driven dealer markets. The eventual multi-agent
  tier uses a per-turn call auction with a uniform crossing price and published
  tie-breaks.
- **Reward:** terminal absolute return multiple, not explicit group-relative
  reward. GRPO supplies the group baseline. P&L is dense, so there is no
  consolation shaping beyond a once-only first-valid-trade bonus.
- **Credit:** v3 sketch only: margin cash loan, maintenance-margin covenant,
  default ends the episode at zero.
- **Adversary:** v2 ships stale-quote pickoff, bait-and-switch depth, and
  trigger-hunting. Each con is scripted and parameterized.
- **Horizon:** 10 turns in v1. Token budget is a wall, not a reward term.
- **Difficulty:** edge width is the first dial. `micro` is intentionally generous:
  visible index, low volatility, and frequent near-crossed quotes.
- **Eval/training:** training-shaped environment with a frozen disjoint-seed eval
  split.

## Reward Rationale

The score marks terminal inventory at the exogenous index:

```text
terminal_return = (cash + inventory * final_index) / (starting_cash + starting_inventory * first_index)
```

`micro` starts the dealer with a small book (5 units). Measured on 09-18:
with an empty book, sell edges (bid above index) are unusable until the
agent first buys, which halves the effective early-episode edge frequency.
Starting inventory makes both edge directions live from turn 1 and is the
thematically correct dealer setup. The denominator is starting net worth,
so passive play still centers on a 1.0 return.

This closes the wash-trading channel. If the agent could mark at its own trade
prices, it could inflate book value by trading with itself or a colluding
counterparty. With an index it cannot trade against, terminal value is outside
the agent's control.

The first-valid-trade bonus is binary and paid once. It exists only to prevent a
zero-trade training collapse where every rollout passively ends at the same
return multiple. It should self-anneal: once all siblings in a GRPO group collect
it, it cancels out of the advantage.

The format bonus is intentionally small and separately logged. It should keep
protocol following visible without becoming the task.

## Pre-Registered Farms

### Passive No-Trade

If all rollouts pass, terminal return is exactly 1.0 and group variance dies.
Mitigation: first-valid-trade bonus and quote settings on `micro` that make
obvious trades available most turns.

### Coin-Flip Gambler

Absolute return under GRPO can reward variance-seeking: a zero-edge gambler can
top a group by chance. Mitigation is measurement, not a hand-written penalty.
`bazaar_env.exploits` compares the coin-flip gambler against the honest dealer
at `micro` and at the hard end. If gambling beats dealing, fix the market design
before publishing.

### Loop Spammer

Models can repeat an invalid action after the state stays unchanged. Mitigation:
named illegal reasons, first illegal action free, and a no-progress stop that
counts only repeated identical illegal/unparseable messages. Legal actions —
including `pass` — reset the window, so buy-and-hold and patient play are never
stopped early (stopping them would re-mark the book at the wrong index).

### Limited-Liability Lottery (v3)

Margin plus default can create a free option: borrow max, gamble, keep upside,
default downside. This is not implemented in v1. It is pre-registered for v3 and
must be tested with a leveraged-gambler policy at high volatility.

## Roadmap

### v1 — Implemented Here

Solo taker, one commodity, visible index, scripted quotes, no credit, no
adversary, 10 turns.

### v2 — Maker + Adversary

The agent posts `bid P Q` and `ask P Q`. Scripted flow arrives with price
sensitivity. Stale quotes get picked off. Bait-and-switch depth and
trigger-hunting arrive here.

### v3 — Credit

Margin loan, maintenance covenant, default, shorting. Detail pass deferred until
the v3 build.

### Multi-Agent

Per-turn call auction with uniform crossing price and published tie-break rules.

## Contamination Note

The frozen eval split is generated from disjoint seeds. The tasks are executable,
not recallable: a price path and quote stream cannot be memorized from the
internet. Contamination risk is mostly implementation leakage, not public-answer
leakage.

## V2 Maker + Adversary (implemented 0.2.0)

V2 turns the agent from a taker into a dealer. The new command is:

```text
quote BP BQ AP AQ
```

The quote replaces the standing market. `pass` leaves the quote standing. The turn order is deliberate and is the adverse-selection lesson:

1. Agent sees `index_t` and posts a quote.
2. The index advances to `index_t+1`.
3. Scripted noise flow fills attractive quotes.
4. Scripted informed flow picks off stale quotes using `index_t+1`.
5. State is marked and returned.

### Implemented adversary cons

1. **Stale-quote pickoff.** If the ask is below the stepped index, informed flow buys from the agent; if the bid is above the stepped index, informed flow sells to the agent. `pickoff_losses` records the loss against the index.
2. **Bait-and-switch depth.** On maker tiers, juicy displayed taker quotes can have lower executable size than displayed size; oversizing returns a named illegal reason.
3. **Trigger-hunting.** The displayed taker quote can move just inside the smallest profitable edge the agent previously accepted. This is scripted and seeded, not an LLM judge.

### V2 scripted anchors (20 seeds, after the 09-18 review fixes)

Review fixes that changed semantics: (1) taker trades now record their
edge, which arms trigger-hunting in real play (it was dead code before);
(2) posted quote size is a hard per-turn exposure cap across all flow --
"up to bid_size" is now true; (3) informed pickoff arrives with
probability 1 - 0.5^pickoff_intensity and takes the remaining stale size
once, instead of looping at full size.

`maker_micro`:

- `tight_quoter`: terminal 0.996, fills 39.3, pickoff losses 5.4 -- overtrades and bleeds, now bounded by posted size.
- `wide_quoter`: terminal 0.999, zero fills -- safe but idle.
- `vol_aware_quoter`: terminal 1.020, fills 17.9, pickoff losses ~0 -- the honest maker anchor.
- `threshold_taker`: terminal 1.027 -- taker edge still exists; the con-3 victim now actually arms the hunt.

`maker_easy`:

- `tight_quoter`: terminal 0.983, pickoff losses 13.7 -- hard-end overtrading still punished.
- `vol_aware_quoter`: terminal 1.002, low fills -- safer but sparse.

These anchors are the v2 exploit catalogue baseline. Model preflights come next;
training is a separate go/no-go.
