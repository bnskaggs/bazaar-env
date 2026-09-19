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

ActionKind = Literal["buy", "sell", "pass", "quote", "borrow", "repay"]

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
    maker: bool = False
    flow_intensity: int = 4
    pickoff_intensity: int = 3
    bait_probability: float = 0.0
    bait_depth_fraction: float = 0.25
    trigger_hunt_probability: float = 0.0
    # v3 credit: a cash margin loan. The rate is the hurdle (borrow only when
    # your edge beats it); max_leverage bounds the blast radius; the covenant
    # plus a one-turn cure window is what closes the limited-liability lottery.
    credit: bool = False
    interest_rate: float = 0.005
    max_leverage: float = 1.0
    maintenance_ratio: float = 0.30


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
    "maker_micro": Tier(starting_inventory=5, vol=0.6, spread=2.0, edge_width=5.0, edge_probability=0.7, maker=True, flow_intensity=4, pickoff_intensity=2, bait_probability=0.15, trigger_hunt_probability=0.15),
    "maker_easy": Tier(starting_inventory=5, vol=1.2, spread=2.5, edge_width=3.0, edge_probability=0.45, maker=True, flow_intensity=5, pickoff_intensity=3, bait_probability=0.25, trigger_hunt_probability=0.25),
    # credit_micro is deliberately capital-poor (cash 300 vs micro's 1000):
    # measured 09-19, with ample cash the loan never binds and credit is pure
    # interest drag - the correct policy would be "never borrow", which
    # teaches nothing. Scarce capital makes "borrow when the edge beats the
    # rate" a real decision.
    # credit_micro's shape is lumpy on purpose (measured 09-19, three tunings):
    # (1) scarce cash (300) or the loan never binds; (2) big displayed sizes
    # (8) or the loan has no marginal value; (3) RARE, FAT edges (p=0.35,
    # width 8) or the opportunity cost of a bank-visit turn dominates and the
    # correct policy is "never borrow". Credit pays when opportunity is lumpy:
    # finance on quiet turns, strike the windfall, repay after.
    "credit_micro": Tier(starting_cash=300.0, starting_inventory=5, vol=0.6, spread=2.0, edge_width=8.0, edge_probability=0.35, max_quote_qty=8, maker=True, flow_intensity=4, pickoff_intensity=2, bait_probability=0.15, trigger_hunt_probability=0.15, credit=True),
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
    source: str = "take"
    edge_vs_index: float = 0.0


@dataclass(frozen=True)
class MarketTask:
    seed: int
    tier: str
    horizon: int
    starting_cash: float
    starting_inventory: int
    index_path: tuple[float, ...]
    quotes: tuple[Quote, ...]
    config: Tier


@dataclass(frozen=True)
class MarketState:
    task: MarketTask
    turn: int = 0
    cash: float = 0.0
    inventory: int = 0
    trades: tuple[Trade, ...] = ()
    standing_quote: Quote | None = None
    first_trade_done: bool = False
    fills: int = 0
    realized_spread: float = 0.0
    pickoff_losses: float = 0.0
    debt: float = 0.0
    interest_paid: float = 0.0
    max_debt: float = 0.0
    margin_call: bool = False
    defaulted: bool = False
    illegal_free_used: bool = False
    stopped: bool = False
    stop_reason: str | None = None


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    quantity: int = 0
    raw: str = ""
    bid: float = 0.0
    bid_size: int = 0
    ask: float = 0.0
    ask_size: int = 0


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
        config=config,
    )


def default_standing_quote(task: MarketTask) -> Quote | None:
    # v2: the agent must choose to make a market. `pass` preserves a standing
    # quote once posted, but there is no default quote at the start.
    return None


def initial_state(task: MarketTask) -> MarketState:
    return MarketState(
        task=task,
        cash=task.starting_cash,
        inventory=task.starting_inventory,
        standing_quote=default_standing_quote(task),
    )


def starting_net_worth(task: MarketTask) -> float:
    return round_money(task.starting_cash + task.starting_inventory * task.index_path[0])


def current_index(state: MarketState) -> float:
    return state.task.index_path[min(state.turn, state.task.horizon)]


def _take_edges(state: MarketState) -> list[tuple[str, float]]:
    return [
        (trade.side, trade.edge_vs_index)
        for trade in state.trades
        if trade.source == "take" and trade.edge_vs_index > 0
    ]


def trigger_threshold(state: MarketState) -> float | None:
    edges = [edge for _, edge in _take_edges(state)]
    return min(edges) if edges else None


def current_quote(state: MarketState) -> Quote:
    base = state.task.quotes[-1] if state.turn >= state.task.horizon else state.task.quotes[state.turn]
    cfg = state.task.config
    if not cfg.maker or not state.trades or cfg.trigger_hunt_probability <= 0:
        return base
    rng = maker_rng(state, 41)
    threshold = trigger_threshold(state)
    if threshold is None or rng.random() > cfg.trigger_hunt_probability:
        return base
    idx = current_index(state)
    bait_edge = max(0.01, threshold - 0.05)
    # Match the most recent profitable take side; bait just inside the inferred
    # trigger and with minimal displayed size.
    side = _take_edges(state)[-1][0]
    if side == "buy":
        return replace(base, ask=round_money(idx - bait_edge), ask_size=max(1, base.ask_size))
    return replace(base, bid=round_money(idx + bait_edge), bid_size=max(1, base.bid_size))


def executable_size(state: MarketState, action: Action, quote: Quote) -> int:
    displayed = quote.ask_size if action.kind == "buy" else quote.bid_size
    cfg = state.task.config
    if not cfg.maker or cfg.bait_probability <= 0:
        return displayed
    idx = current_index(state)
    edge = (idx - quote.ask) if action.kind == "buy" else (quote.bid - idx)
    if edge <= 0:
        return displayed
    rng = maker_rng(state, 73)
    if rng.random() > cfg.bait_probability:
        return displayed
    return max(1, int(displayed * cfg.bait_depth_fraction))


def marked_net_worth(state: MarketState) -> float:
    return round_money(state.cash + state.inventory * current_index(state))


def equity(state: MarketState) -> float:
    """Net worth after debt — the number the terminal score is built on."""
    return round_money(marked_net_worth(state) - state.debt)


def max_borrowable(state: MarketState) -> float:
    """Remaining borrow capacity: total debt may not exceed max_leverage x
    current equity (borrowing itself is equity-neutral: +cash, +debt)."""
    cap = state.task.config.max_leverage * equity(state)
    return round_money(max(0.0, cap - state.debt))


def in_covenant_breach(state: MarketState) -> bool:
    if state.debt <= 0:
        return False
    return equity(state) < state.task.config.maintenance_ratio * state.debt


def mark_to_index_pnl(state: MarketState) -> float:
    return round_money(equity(state) - starting_net_worth(state.task))


def is_done(state: MarketState) -> bool:
    return state.stopped or state.turn >= state.task.horizon


def public_state(state: MarketState, recent: int = 3) -> dict:
    quote = current_quote(state)
    result = {
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
    if state.task.config.maker:
        # Maker-only keys: keep the v1 prompt format byte-identical to what the
        # v1 training run saw (frozen-format discipline for comparability).
        result["standing_quote"] = (
            None
            if state.standing_quote is None
            else {
                "bid": state.standing_quote.bid,
                "bid_size": state.standing_quote.bid_size,
                "ask": state.standing_quote.ask,
                "ask_size": state.standing_quote.ask_size,
            }
        )
        result["fills"] = state.fills
        result["realized_spread"] = round_money(state.realized_spread)
        result["pickoff_losses"] = round_money(state.pickoff_losses)
    if state.task.config.credit:
        # Credit-only keys, same prompt-format discipline as the maker keys.
        result["debt"] = round_money(state.debt)
        result["equity"] = equity(state)
        result["borrowable"] = max_borrowable(state)
        result["margin_call"] = state.margin_call
    return result


def render_state(state: MarketState) -> str:
    return json.dumps(public_state(state), sort_keys=True)


def legal_quantity(action: Action) -> bool:
    return action.quantity > 0 and float(action.quantity).is_integer()


def illegal_reason(state: MarketState, action: Action) -> str | None:
    if is_done(state):
        return "episode is already complete"
    if action.kind == "pass":
        return None
    if action.kind == "quote":
        if not state.task.config.maker:
            return "quote action is only legal on maker tiers"
        if action.bid <= 0 or action.ask <= 0:
            return "quote prices must be positive"
        if action.bid >= action.ask:
            return "bid must be below ask"
        if action.bid_size <= 0 or action.ask_size <= 0:
            return "quote sizes must be positive integers"
        return None
    if not legal_quantity(action):
        return "quantity must be a positive integer"

    quote = current_quote(state)
    if action.kind == "buy":
        ask_size = executable_size(state, action, quote)
        if action.quantity > ask_size:
            if ask_size < quote.ask_size:
                return f"displayed ask size {quote.ask_size} but executable size is {ask_size}"
            return f"quantity {action.quantity} exceeds ask size {quote.ask_size}"
        cost = quote.ask * action.quantity
        buying_power = state.cash + (max_borrowable(state) if state.task.config.credit else 0.0)
        if cost > buying_power + 1e-9:
            if state.task.config.credit:
                return (
                    f"cost {round_money(cost)} exceeds buying power "
                    f"{round_money(buying_power)} (cash + margin capacity)"
                )
            return f"not enough cash to buy {action.quantity} at ask {quote.ask}"
        return None
    if action.kind == "sell":
        bid_size = executable_size(state, action, quote)
        if action.quantity > bid_size:
            if bid_size < quote.bid_size:
                return f"displayed bid size {quote.bid_size} but executable size is {bid_size}"
            return f"quantity {action.quantity} exceeds bid size {quote.bid_size}"
        if action.quantity > state.inventory:
            return f"not enough inventory to sell {action.quantity}"
        return None
    if action.kind in {"borrow", "repay"}:
        # Margin is automatic (measured 09-19: explicit bank-visit turns can
        # never pay for themselves at a 10-turn horizon). The verbs stay
        # parseable so models get an instructive correction.
        if state.task.config.credit:
            return (
                "margin is automatic on this tier: buying beyond your cash "
                "draws the loan (within leverage), and surplus cash repays it "
                "at the end of each turn"
            )
        return "there is no credit on this tier"
    return "unknown action"


def settle_turn(state: MarketState) -> tuple[MarketState, list[str]]:
    """Advance the world by one turn: index steps, flow resolves against any
    standing quote, interest accrues, and the covenant is enforced. Every
    consumed turn — trade, quote, pass, bank visit, or wasted illegal — goes
    through here: the market never stops.
    """
    notes: list[str] = []
    state = replace(state, turn=min(state.turn + 1, state.task.horizon))
    state, flow_notes = flow_against_quote(state)
    notes.extend(flow_notes)

    if state.task.config.credit and state.debt > 0 and state.cash > 0:
        # Surplus cash repays the loan automatically: idle debt cannot exist,
        # which structurally deletes the idle-borrower farm.
        repay = round_money(min(state.cash, state.debt))
        state = replace(
            state,
            cash=round_money(state.cash - repay),
            debt=round_money(state.debt - repay),
        )
        notes.append(f"auto-repaid {repay} (debt {round_money(state.debt)})")

    if state.task.config.credit and state.debt > 0:
        interest = round_money(state.debt * state.task.config.interest_rate)
        if interest > 0:
            state = replace(
                state,
                debt=round_money(state.debt + interest),
                interest_paid=round_money(state.interest_paid + interest),
            )
            notes.append(f"interest {interest} accrued (debt {round_money(state.debt)})")
        state = replace(state, max_debt=max(state.max_debt, state.debt))

        if in_covenant_breach(state):
            if state.margin_call:
                state = replace(
                    state,
                    defaulted=True,
                    stopped=True,
                    stop_reason="margin call not cured: default",
                )
                notes.append("DEFAULT: margin call not cured; episode over")
            else:
                state = replace(state, margin_call=True)
                notes.append(
                    "MARGIN CALL: equity below maintenance; cure by end of next "
                    "turn (repay or sell down) or default"
                )
        elif state.margin_call:
            state = replace(state, margin_call=False)
            notes.append("margin call cured")
    elif state.margin_call:
        # Debt fully repaid (or zero): breach is impossible, clear the flag.
        state = replace(state, margin_call=False)
        notes.append("margin call cured")
    return state, notes


def apply_action(state: MarketState, action: Action) -> StepResult:
    reason = illegal_reason(state, action)
    if reason is not None:
        if state.illegal_free_used:
            nxt, notes = settle_turn(state)
            turn_note = "Wasted a turn."
            if notes:
                turn_note += " " + "; ".join(notes)
        else:
            nxt = replace(state, illegal_free_used=True)
            turn_note = "No turn consumed; further illegal actions waste a turn."
        return StepResult(
            state=nxt,
            reply=f"illegal: {action.raw or action.kind} - {reason}. {turn_note}",
            legal=False,
        )

    if state.task.config.maker and action.kind in {"pass", "quote"}:
        return apply_maker_action(state, action)

    if action.kind == "pass":
        nxt, notes = settle_turn(state)
        reply = "ok: passed"
        if notes:
            reply += "; " + "; ".join(notes)
        return StepResult(state=nxt, reply=reply, legal=True)

    quote = current_quote(state)
    index_value = current_index(state)
    price = quote.ask if action.kind == "buy" else quote.bid
    quantity = action.quantity
    cash_delta = -price * quantity if action.kind == "buy" else price * quantity
    inventory_delta = quantity if action.kind == "buy" else -quantity
    take_edge = round_money(
        (index_value - price) if action.kind == "buy" else (price - index_value)
    )
    trade = Trade(
        turn=state.turn + 1,
        side=action.kind,
        quantity=quantity,
        price=price,
        index_value=index_value,
        source="take",
        edge_vs_index=take_edge,
    )
    new_cash = round_money(state.cash + cash_delta)
    new_debt = state.debt
    margin_note = ""
    if new_cash < 0:
        # Auto-margin: the broker fronts the shortfall (legality already
        # checked cost against cash + capacity).
        draw = round_money(-new_cash)
        new_debt = round_money(state.debt + draw)
        new_cash = 0.0
        margin_note = f" (margin draw {draw}, debt {new_debt})"
    nxt = replace(
        state,
        cash=new_cash,
        debt=new_debt,
        inventory=state.inventory + inventory_delta,
        trades=state.trades + (trade,),
        first_trade_done=True,
    )
    nxt, notes = settle_turn(nxt)
    reply = f"ok: {action.kind} {quantity} @ {price} (edge_vs_index={take_edge}){margin_note}"
    if notes:
        reply += "; " + "; ".join(notes)
    return StepResult(state=nxt, reply=reply, legal=True, trade=trade)


def _fill(state: MarketState, side: Literal["buy", "sell"], quantity: int, price: float, source: str, index_value: float) -> tuple[MarketState, Trade | None]:
    if quantity <= 0:
        return state, None
    if side == "buy":
        buying_power = state.cash + (
            max_borrowable(state) if state.task.config.credit else 0.0
        )
        quantity = min(quantity, int(buying_power // price)) if price > 0 else 0
        if quantity <= 0:
            return state, None
        cash_delta = -price * quantity
        inventory_delta = quantity
        edge = index_value - price
    else:
        quantity = min(quantity, state.inventory)
        if quantity <= 0:
            return state, None
        cash_delta = price * quantity
        inventory_delta = -quantity
        edge = price - index_value
    trade = Trade(
        # Callers advance the turn before flow resolves, so state.turn is
        # already the 1-based turn this fill belongs to.
        turn=state.turn,
        side=side,
        quantity=quantity,
        price=round_money(price),
        index_value=index_value,
        source=source,
        edge_vs_index=round_money(edge),
    )
    realized = max(0.0, edge * quantity)
    pickoff = max(0.0, -edge * quantity) if source == "pickoff" else 0.0
    new_cash = round_money(state.cash + cash_delta)
    new_debt = state.debt
    if new_cash < 0:
        # Auto-margin covers fills on the agent's own bid, within the leverage
        # cap already enforced by the buying-power clamp above.
        new_debt = round_money(state.debt - new_cash)
        new_cash = 0.0
    return replace(
        state,
        cash=new_cash,
        debt=new_debt,
        inventory=state.inventory + inventory_delta,
        trades=state.trades + (trade,),
        first_trade_done=True,
        fills=state.fills + quantity,
        realized_spread=round_money(state.realized_spread + realized),
        pickoff_losses=round_money(state.pickoff_losses + pickoff),
    ), trade


def maker_rng(state: MarketState, salt: int) -> random.Random:
    return random.Random(state.task.seed * 10_000 + state.turn * 101 + salt)


def flow_against_quote(state: MarketState) -> tuple[MarketState, list[str]]:
    quote = state.standing_quote
    if quote is None:
        return state, []
    cfg = state.task.config
    idx = current_index(state)
    notes: list[str] = []
    # Posted size is the per-turn exposure cap, per side, across ALL flow.
    # Without this, "up to bid_size" printed in the rules would be a lie and
    # sizing would be meaningless as a risk control.
    bid_remaining = quote.bid_size
    ask_remaining = quote.ask_size

    # Noise flow: reservation prices around the new index. Sellers hit our bid;
    # buyers lift our ask. Tight quotes fill, wide quotes idle.
    rng = maker_rng(state, 17)
    for _ in range(cfg.flow_intensity):
        if rng.random() < 0.5:
            reservation = idx - rng.uniform(-cfg.spread, cfg.edge_width)
            if quote.bid >= reservation and bid_remaining > 0:
                state, trade = _fill(state, "buy", 1, quote.bid, "noise", idx)
                if trade:
                    bid_remaining -= trade.quantity
                    notes.append(f"noise_seller_fill buy 1 @ {quote.bid}")
        else:
            reservation = idx + rng.uniform(-cfg.spread, cfg.edge_width)
            if quote.ask <= reservation and ask_remaining > 0:
                state, trade = _fill(state, "sell", 1, quote.ask, "noise", idx)
                if trade:
                    ask_remaining -= trade.quantity
                    notes.append(f"noise_buyer_fill sell 1 @ {quote.ask}")

    # Informed flow: picks off whatever stale size remains after noise.
    # pickoff_intensity = number of informed traders; each notices a stale
    # quote with an independent seeded coin, and any one of them takes the
    # full remaining size. Arrival probability = 1 - 0.5**intensity.
    pickoff_rng = maker_rng(state, 29)
    informed_arrives = any(
        pickoff_rng.random() < 0.5 for _ in range(cfg.pickoff_intensity)
    )
    if informed_arrives:
        if quote.ask < idx and ask_remaining > 0:
            state, trade = _fill(state, "sell", ask_remaining, quote.ask, "pickoff", idx)
            if trade:
                ask_remaining -= trade.quantity
                notes.append(f"pickoff sell {trade.quantity} @ {quote.ask}")
        if quote.bid > idx and bid_remaining > 0:
            state, trade = _fill(state, "buy", bid_remaining, quote.bid, "pickoff", idx)
            if trade:
                bid_remaining -= trade.quantity
                notes.append(f"pickoff buy {trade.quantity} @ {quote.bid}")
    return state, notes


def apply_maker_action(state: MarketState, action: Action) -> StepResult:
    if action.kind == "quote":
        standing = Quote(
            bid=round_money(action.bid),
            bid_size=action.bid_size,
            ask=round_money(action.ask),
            ask_size=action.ask_size,
        )
        state = replace(state, standing_quote=standing)
        reply = f"ok: quote bid {standing.bid} x {standing.bid_size} / ask {standing.ask} x {standing.ask_size}"
    else:
        reply = "ok: standing quote left unchanged"
    # Quote at index_t, then the world moves to index_t+1 before fills;
    # settle_turn also accrues interest and enforces the covenant.
    state, notes = settle_turn(state)
    if notes:
        reply += "; " + "; ".join(notes)
    return StepResult(state=state, reply=reply, legal=True)


def reward_components(state: MarketState, format_ok: bool) -> RewardComponents:
    if state.defaulted:
        # Default wipes the terminal component. Strictly dominated by passive
        # play (~1.0), which is what keeps borrow-and-pray a losing strategy.
        terminal_return = 0.0
    else:
        terminal_return = equity(state) / starting_net_worth(state.task)
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
ACTION_RE = re.compile(
    r"\b(?:(quote)\s+([0-9]+(?:\.[0-9]+)?)\s+(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s+(\d+)|(buy|sell|borrow|repay)\s+(\d+)|(pass))\b",
    re.IGNORECASE,
)


def parse_action(text: str) -> Action | None:
    match = None
    for match in ACTION_RE.finditer(text):
        pass
    if match is None:
        return None
    if match.group(8) is not None:
        return Action(kind="pass", raw="pass")
    if match.group(1) is not None:
        bid = float(match.group(2))
        bid_size = int(match.group(3))
        ask = float(match.group(4))
        ask_size = int(match.group(5))
        return Action(
            kind="quote",
            bid=bid,
            bid_size=bid_size,
            ask=ask,
            ask_size=ask_size,
            raw=f"quote {bid:g} {bid_size} {ask:g} {ask_size}",
        )
    kind: ActionKind = match.group(6).lower()  # buy | sell | borrow | repay
    quantity = int(match.group(7))
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


def quote_action(index_value: float, width: float, size: int = 3) -> Action:
    bid = round_money(index_value - width / 2)
    ask = round_money(index_value + width / 2)
    return Action(
        kind="quote",
        bid=bid,
        bid_size=size,
        ask=ask,
        ask_size=size,
        raw=f"quote {bid:g} {size} {ask:g} {size}",
    )


def wide_quoter_policy(state: MarketState) -> Action:
    return quote_action(current_index(state), width=12.0, size=3)


def tight_quoter_policy(state: MarketState) -> Action:
    return quote_action(current_index(state), width=0.5, size=5)


def vol_aware_quoter_policy(state: MarketState) -> Action:
    # Simple anchor: quote roughly 3 sigma wide, with a floor. It should trade
    # enough to earn flow while avoiding most pickoffs.
    width = max(1.5, 3.0 * state.task.config.vol + state.task.config.spread)
    return quote_action(current_index(state), width=width, size=3)


def buying_power(state: MarketState) -> float:
    return round_money(
        state.cash + (max_borrowable(state) if state.task.config.credit else 0.0)
    )


def margin_taker_policy(threshold: float = 1.0) -> Policy:
    """The credit anchor: size takes to full buying power (cash + margin) when
    the edge clears the threshold, and let auto-repay handle the loan. The
    skill the tier teaches: leverage into edges, not into hope."""

    def policy(state: MarketState) -> Action:
        quote = current_quote(state)
        idx = current_index(state)
        if idx - quote.ask >= threshold:
            qty = min(quote.ask_size, int(buying_power(state) // quote.ask))
            if qty > 0:
                return Action(kind="buy", quantity=qty, raw=f"buy {qty}")
        if quote.bid - idx >= threshold and state.inventory > 0:
            qty = min(quote.bid_size, state.inventory)
            return Action(kind="sell", quantity=qty, raw=f"sell {qty}")
        return passive_policy(state)

    return policy


def margin_gambler_policy(seed: int = 0) -> Policy:
    # The limited-liability lottery, auto-margin edition: max-size coin-flip
    # trades regardless of edge. Covenant + interest must make this lose.
    rng = random.Random(seed)

    def policy(state: MarketState) -> Action:
        quote = current_quote(state)
        if rng.random() < 0.5:
            qty = min(quote.ask_size, max(1, int(buying_power(state) // quote.ask)))
            if qty > 0 and buying_power(state) >= quote.ask:
                return Action(kind="buy", quantity=qty, raw=f"buy {qty}")
        if state.inventory > 0:
            qty = min(quote.bid_size, state.inventory)
            return Action(kind="sell", quantity=qty, raw=f"sell {qty}")
        return passive_policy(state)

    return policy


def threshold_taker_policy(threshold: float = 1.5) -> Policy:
    def policy(state: MarketState) -> Action:
        quote = current_quote(state)
        idx = current_index(state)
        if quote.ask <= idx - threshold and state.cash >= quote.ask:
            qty = min(quote.ask_size, max(1, int(state.cash // quote.ask)))
            return Action(kind="buy", quantity=qty, raw=f"buy {qty}")
        if quote.bid >= idx + threshold and state.inventory > 0:
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
        "fills": result.final_state.fills,
        "realized_spread": result.final_state.realized_spread,
        "pickoff_losses": result.final_state.pickoff_losses,
        "final_equity": equity(result.final_state),
        "max_debt": result.final_state.max_debt,
        "interest_paid": result.final_state.interest_paid,
        "defaulted": result.final_state.defaulted,
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







