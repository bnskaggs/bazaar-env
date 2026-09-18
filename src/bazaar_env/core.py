"""Framework-free Bazaar market engine.

The adapter is intentionally thin. This module owns the deterministic market
rules: index generation, scripted quotes, execution, marking, rewards, and
scripted policies used for exploit probes.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Literal

ActionKind = Literal["buy", "sell", "pass"]

FORMAT_BONUS = 0.2
FIRST_TRADE_BONUS = 0.05


@dataclass(frozen=True)
class Tier:
    horizon: int = 10
    starting_cash: float = 1_000.0
    starting_inventory: int = 0
    start_index: float = 100.0
    vol: float = 1.0
    spread: float = 2.0
    edge_width: float = 4.0
    edge_probability: float = 0.75
    max_quote_qty: int = 5


TIERS: dict[str, Tier] = {
    # The band hypothesis: a 7B-9B model should often recognize and take
    # obvious mispricings while still making sizing/horizon mistakes.
    # micro starts with a dealer book (inventory > 0) so sell edges are
    # usable from turn 1 — measured 09-18: with an empty book the effective
    # early-episode edge frequency is about half the nominal one.
    "micro": Tier(starting_inventory=5, vol=0.6, spread=2.0, edge_width=5.0, edge_probability=0.9),
    "trivial": Tier(vol=1.0, spread=2.0, edge_width=3.0, edge_probability=0.65),
    "easy": Tier(vol=1.5, spread=3.0, edge_width=2.0, edge_probability=0.45),
    "hard": Tier(vol=3.0, spread=4.0, edge_width=1.0, edge_probability=0.2),
}


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    bid_size: int
    ask_size: int


@dataclass(frozen=True)
class Trade:
    turn: int
    side: Literal["buy", "sell"]
    quantity: int
    price: float
    index_value: float


@dataclass(frozen=True)
class MarketTask:
    seed: int
    tier: str
    horizon: int
    starting_cash: float
    starting_inventory: int
    index_path: tuple[float, ...]
    quotes: tuple[Quote, ...]


@dataclass(frozen=True)
class MarketState:
    task: MarketTask
    turn: int = 0
    cash: float = 0.0
    inventory: int = 0
    trades: tuple[Trade, ...] = ()
    first_trade_done: bool = False
    illegal_free_used: bool = False
    stopped: bool = False
    stop_reason: str | None = None


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    quantity: int = 0
    raw: str = ""


@dataclass(frozen=True)
class RewardComponents:
    terminal_return: float
    first_trade_bonus: float
    format: float

    @property
    def total(self) -> float:
        return self.terminal_return + self.first_trade_bonus + self.format


@dataclass(frozen=True)
class StepResult:
    state: MarketState
    reply: str
    legal: bool
    trade: Trade | None = None


@dataclass(frozen=True)
class ReplayResult:
    final_state: MarketState
    assistant_turns: int
    parsed_turns: int
    illegal_actions: int
    repeated_messages: int
    no_progress_stopped: bool
    num_trades: int
    first_trade_turn: int | None
    terminal_return: float
    reward: float
    components: dict[str, float]


def round_money(value: float) -> float:
    return round(value + 1e-12, 2)


def tier_config(tier: str, **overrides) -> Tier:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(TIERS)}")
    config = TIERS[tier]
    valid = set(Tier.__dataclass_fields__)
    clean = {key: value for key, value in overrides.items() if value is not None}
    unknown = set(clean) - valid
    if unknown:
        raise ValueError(f"unknown tier override(s): {sorted(unknown)}")
    return replace(config, **clean)


def build_index_path(rng: random.Random, config: Tier) -> tuple[float, ...]:
    values = [config.start_index]
    for _ in range(config.horizon):
        # Additive bounded random walk: legible numbers, no accidental negative
        # prices, enough movement for inventory risk.
        nxt = max(1.0, values[-1] + rng.gauss(0.0, config.vol))
        values.append(round_money(nxt))
    return tuple(values)


def build_quote(rng: random.Random, index_value: float, config: Tier) -> Quote:
    half_spread = config.spread / 2
    bid = index_value - half_spread - abs(rng.gauss(0.0, config.spread / 6))
    ask = index_value + half_spread + abs(rng.gauss(0.0, config.spread / 6))

    if rng.random() < config.edge_probability:
        if rng.random() < 0.5:
            # Buy edge: the counterparty is willing to sell below reference.
            ask = index_value - rng.uniform(0.5, config.edge_width)
        else:
            # Sell edge: the counterparty is willing to buy above reference.
            bid = index_value + rng.uniform(0.5, config.edge_width)

    bid = max(0.01, round_money(bid))
    ask = max(0.01, round_money(ask))
    return Quote(
        bid=bid,
        ask=ask,
        bid_size=rng.randint(1, config.max_quote_qty),
        ask_size=rng.randint(1, config.max_quote_qty),
    )


def generate(tier: str = "micro", seed: int = 0, **overrides) -> MarketTask:
    config = tier_config(tier, **overrides)
    rng = random.Random(seed)
    index_path = build_index_path(rng, config)
    quotes = tuple(build_quote(rng, index_path[turn], config) for turn in range(config.horizon))
    return MarketTask(
        seed=seed,
        tier=tier,
        horizon=config.horizon,
        starting_cash=config.starting_cash,
        starting_inventory=config.starting_inventory,
        index_path=index_path,
        quotes=quotes,
    )


def initial_state(task: MarketTask) -> MarketState:
    return MarketState(task=task, cash=task.starting_cash, inventory=task.starting_inventory)


def starting_net_worth(task: MarketTask) -> float:
    return round_money(task.starting_cash + task.starting_inventory * task.index_path[0])


def current_index(state: MarketState) -> float:
    return state.task.index_path[min(state.turn, state.task.horizon)]


def current_quote(state: MarketState) -> Quote:
    if state.turn >= state.task.horizon:
        return state.task.quotes[-1]
    return state.task.quotes[state.turn]


def marked_net_worth(state: MarketState) -> float:
    return round_money(state.cash + state.inventory * current_index(state))


def mark_to_index_pnl(state: MarketState) -> float:
    return round_money(marked_net_worth(state) - starting_net_worth(state.task))


def is_done(state: MarketState) -> bool:
    return state.stopped or state.turn >= state.task.horizon


def public_state(state: MarketState, recent: int = 3) -> dict:
    quote = current_quote(state)
    return {
        "turn": state.turn + 1 if not is_done(state) else state.task.horizon,
        "turns_remaining": max(0, state.task.horizon - state.turn),
        "cash": round_money(state.cash),
        "inventory": state.inventory,
        "index_value": current_index(state),
        "mark_to_index_pnl": mark_to_index_pnl(state),
        "quotes": {
            "bid": quote.bid,
            "bid_size": quote.bid_size,
            "ask": quote.ask,
            "ask_size": quote.ask_size,
        },
        "recent_trades": [
            {
                "turn": trade.turn,
                "side": trade.side,
                "quantity": trade.quantity,
                "price": trade.price,
                "index_value": trade.index_value,
            }
            for trade in state.trades[-recent:]
        ],
    }


def render_state(state: MarketState) -> str:
    return json.dumps(public_state(state), sort_keys=True)


def legal_quantity(action: Action) -> bool:
    return action.quantity > 0 and float(action.quantity).is_integer()


def illegal_reason(state: MarketState, action: Action) -> str | None:
    if is_done(state):
        return "episode is already complete"
    if action.kind == "pass":
        return None
    if not legal_quantity(action):
        return "quantity must be a positive integer"

    quote = current_quote(state)
    if action.kind == "buy":
        if action.quantity > quote.ask_size:
            return f"quantity {action.quantity} exceeds ask size {quote.ask_size}"
        cost = quote.ask * action.quantity
        if cost > state.cash + 1e-9:
            return f"not enough cash to buy {action.quantity} at ask {quote.ask}"
        return None
    if action.kind == "sell":
        if action.quantity > quote.bid_size:
            return f"quantity {action.quantity} exceeds bid size {quote.bid_size}"
        if action.quantity > state.inventory:
            return f"not enough inventory to sell {action.quantity}"
        return None
    return "unknown action"


def apply_action(state: MarketState, action: Action) -> StepResult:
    reason = illegal_reason(state, action)
    if reason is not None:
        if state.illegal_free_used:
            nxt = replace(state, turn=min(state.turn + 1, state.task.horizon))
            turn_note = "Wasted a turn."
        else:
            nxt = replace(state, illegal_free_used=True)
            turn_note = "No turn consumed; further illegal actions waste a turn."
        return StepResult(
            state=nxt,
            reply=f"illegal: {action.raw or action.kind} - {reason}. {turn_note}",
            legal=False,
        )

    if action.kind == "pass":
        return StepResult(
            state=replace(state, turn=min(state.turn + 1, state.task.horizon)),
            reply="ok: passed",
            legal=True,
        )

    quote = current_quote(state)
    index_value = current_index(state)
    price = quote.ask if action.kind == "buy" else quote.bid
    quantity = action.quantity
    cash_delta = -price * quantity if action.kind == "buy" else price * quantity
    inventory_delta = quantity if action.kind == "buy" else -quantity
    trade = Trade(
        turn=state.turn + 1,
        side=action.kind,
        quantity=quantity,
        price=price,
        index_value=index_value,
    )
    nxt = replace(
        state,
        turn=min(state.turn + 1, state.task.horizon),
        cash=round_money(state.cash + cash_delta),
        inventory=state.inventory + inventory_delta,
        trades=state.trades + (trade,),
        first_trade_done=True,
    )
    edge = round_money((index_value - price) if action.kind == "buy" else (price - index_value))
    return StepResult(
        state=nxt,
        reply=f"ok: {action.kind} {quantity} @ {price} (edge_vs_index={edge})",
        legal=True,
        trade=trade,
    )


def reward_components(state: MarketState, format_ok: bool) -> RewardComponents:
    terminal_return = marked_net_worth(state) / starting_net_worth(state.task)
    return RewardComponents(
        terminal_return=terminal_return,
        first_trade_bonus=FIRST_TRADE_BONUS if state.first_trade_done else 0.0,
        format=FORMAT_BONUS if format_ok else 0.0,
    )


def summarize_reward(state: MarketState, format_ok: bool) -> dict[str, float]:
    components = reward_components(state, format_ok)
    return {
        "terminal_return": components.terminal_return,
        "first_trade_bonus": components.first_trade_bonus,
        "format": components.format,
    }


# Lenient extraction, last match wins — the proven Magic Sort protocol.
# Requiring the whole message to be a bare command punishes chatty models for
# protocol, not planning (the seg-4 confound).
ACTION_RE = re.compile(r"\b(?:(buy|sell)\s+(\d+)|(pass))\b", re.IGNORECASE)


def parse_action(text: str) -> Action | None:
    match = None
    for match in ACTION_RE.finditer(text):
        pass
    if match is None:
        return None
    if match.group(3) is not None:
        return Action(kind="pass", raw="pass")
    kind: ActionKind = "buy" if match.group(1).lower() == "buy" else "sell"
    quantity = int(match.group(2))
    return Action(kind=kind, quantity=quantity, raw=f"{kind} {quantity}")


Policy = Callable[[MarketState], Action]


def passive_policy(state: MarketState) -> Action:
    return Action(kind="pass", raw="pass")


def honest_dealer_policy(state: MarketState) -> Action:
    quote = current_quote(state)
    index_value = current_index(state)
    if quote.ask < index_value and state.cash >= quote.ask:
        qty = min(quote.ask_size, int(state.cash // quote.ask))
        if qty > 0:
            return Action(kind="buy", quantity=qty, raw=f"buy {qty}")
    if quote.bid > index_value and state.inventory > 0:
        qty = min(quote.bid_size, state.inventory)
        return Action(kind="sell", quantity=qty, raw=f"sell {qty}")
    return passive_policy(state)


def coin_flip_gambler_policy(seed: int = 0) -> Policy:
    rng = random.Random(seed)

    def policy(state: MarketState) -> Action:
        quote = current_quote(state)
        if rng.random() < 0.5 and state.cash >= quote.ask:
            qty = min(quote.ask_size, max(1, int(state.cash // quote.ask)))
            return Action(kind="buy", quantity=qty, raw=f"buy {qty}")
        if state.inventory > 0:
            qty = min(quote.bid_size, state.inventory)
            return Action(kind="sell", quantity=qty, raw=f"sell {qty}")
        return passive_policy(state)

    return policy


def loop_spammer_policy(state: MarketState) -> Action:
    return Action(kind="sell", quantity=999, raw="sell 999")


def run_policy(task: MarketTask, policy: Policy, repeat_stop: int = 4) -> ReplayResult:
    state = initial_state(task)
    messages: list[str] = []
    illegal_actions = 0
    parsed_turns = 0
    repeated_messages = 0
    no_progress_stopped = False
    # The no-progress stop counts only *illegal* repeats. A legal action
    # (including pass) advances the turn, so it is progress by definition;
    # stopping repeated legal passes would break buy-and-hold play.
    bad_message: str | None = None
    bad_streak = 0
    while not is_done(state):
        action = policy(state)
        messages.append(action.raw)
        parsed_turns += 1
        result = apply_action(state, action)
        if not result.legal:
            illegal_actions += 1
            if action.raw == bad_message:
                bad_streak += 1
            else:
                bad_message = action.raw
                bad_streak = 1
        else:
            bad_message = None
            bad_streak = 0
        state = result.state
        if bad_streak >= repeat_stop:
            repeated_messages += 1
            no_progress_stopped = True
            state = replace(state, stopped=True, stop_reason=f"repeated illegal action {repeat_stop} times")
            break
    components = summarize_reward(state, format_ok=True)
    return ReplayResult(
        final_state=state,
        assistant_turns=len(messages),
        parsed_turns=parsed_turns,
        illegal_actions=illegal_actions,
        repeated_messages=repeated_messages,
        no_progress_stopped=no_progress_stopped,
        num_trades=len(state.trades),
        first_trade_turn=None if not state.trades else state.trades[0].turn,
        terminal_return=components["terminal_return"],
        reward=sum(components.values()),
        components=components,
    )


def result_to_dict(result: ReplayResult) -> dict:
    return {
        "reward": result.reward,
        "terminal_return": result.terminal_return,
        "num_trades": result.num_trades,
        "first_trade_turn": result.first_trade_turn,
        "illegal_actions": result.illegal_actions,
        "no_progress_stopped": result.no_progress_stopped,
        "final_net_worth": marked_net_worth(result.final_state),
        "components": result.components,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="micro")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=["passive", "honest", "gambler", "spammer"], default="honest")
    args = parser.parse_args(list(argv) if argv is not None else None)
    policies: dict[str, Policy] = {
        "passive": passive_policy,
        "honest": honest_dealer_policy,
        "gambler": coin_flip_gambler_policy(args.seed),
        "spammer": loop_spammer_policy,
    }
    task = generate(tier=args.tier, seed=args.seed)
    result = run_policy(task, policies[args.policy])
    print(json.dumps(result_to_dict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
