"""Verifiers adapter for Bazaar.

The market logic lives in ``core.py``. This module builds seeded datasets, runs
the multi-turn text protocol, and replays transcripts for server-authoritative
scoring.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import verifiers as vf
from datasets import Dataset

from . import core

RULES = (
    "You are trading one commodity in Bazaar, a deterministic dealer-market "
    "environment. Each turn you see JSON state. `index_value` is the fair "
    "reference price used to mark terminal net worth. `quotes.bid` is the price "
    "where the counterparty will buy from you, up to `bid_size`. `quotes.ask` "
    "is the price where the counterparty will sell to you, up to `ask_size`. "
    "You may start with inventory already in your book. Terminal score is final "
    "net worth marked at the final index, divided by your starting net worth "
    "(cash plus starting inventory marked at the first index), plus a small "
    "once-only bonus for executing a real trade and "
    "a small format bonus. Reply each turn with one command: `buy Q`, "
    "`sell Q`, `pass`, or (on maker tiers) `quote BP BQ AP AQ`, where "
    "BP/BQ are your bid price/size and AP/AQ are your ask price/size. "
    "A quote replaces your standing market; `pass` leaves it unchanged. "
    "You cannot short: sell quantity is capped by inventory, and "
    "buy quantity is capped by cash. There are {horizon} turns."
)

CREDIT_RULES = (
    " This tier has an automatic margin loan: buying beyond your cash draws "
    "the loan for you, and surplus cash repays it at the end of each turn. "
    "Interest accrues on outstanding debt at {rate_pct}% per turn. Total "
    "debt may not exceed {max_leverage}x your equity (equity = cash + "
    "inventory marked at the index - debt), which caps your buying power. "
    "COVENANT: if equity falls below {maintenance_pct}% of debt you get a "
    "MARGIN CALL and must cure it (sell down) by the end of the next turn, "
    "or you DEFAULT: the episode ends and your terminal score is 0. Your "
    "terminal score is final equity divided by starting net worth."
)

# Optional rules addendum: states the trading atom explicitly. Instruction
# explicitness is a difficulty knob — with the hint the task measures
# execution (sizing, timing, inventory); without it, strategy discovery.
STRATEGY_HINT = (
    " Hint: a quote is mispriced when the ask is below index_value (cheap to "
    "buy) or the bid is above index_value (rich to sell). You profit by "
    "trading mispriced quotes and unwinding near fair value; trading at "
    "prices worse than index_value loses money at marking."
)


def assistant_texts(completion: list[dict[str, Any]]) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in completion
        if message.get("role") == "assistant"
    ]


def task_from_info(info: dict[str, Any]) -> core.MarketTask:
    quotes = tuple(
        core.Quote(
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            bid_size=int(row["bid_size"]),
            ask_size=int(row["ask_size"]),
        )
        for row in info["quotes"]
    )
    cfg = info.get("config") or {}
    config = core.tier_config(str(info["tier"]), **cfg)
    return core.MarketTask(
        seed=int(info["seed"]),
        tier=str(info["tier"]),
        horizon=int(info["horizon"]),
        starting_cash=float(info["starting_cash"]),
        starting_inventory=int(info.get("starting_inventory", 0)),
        index_path=tuple(float(value) for value in info["index_path"]),
        quotes=quotes,
        config=config,
    )


def replay(info: dict[str, Any], completion: list[dict[str, Any]]) -> core.ReplayResult:
    """Replay a model transcript from the authoritative starting task."""

    task = task_from_info(info)
    state = core.initial_state(task)
    assistant = assistant_texts(completion)
    assistant_turns = len(assistant)
    parsed_turns = 0
    illegal_actions = 0
    repeated_messages = 0
    no_progress_stopped = False
    parse_warnings = 0
    repeat_stop = int(info.get("repeat_stop", 4))
    # No-progress streak: only illegal/unparseable outcomes count; any legal
    # action (including pass) is progress and resets the window.
    bad_message: str | None = None
    bad_streak = 0

    for text in assistant:
        normalized = " ".join(text.lower().split())

        action = core.parse_action(text)
        if action is None:
            if parse_warnings == 0:
                parse_warnings += 1
            else:
                illegal_actions += 1
                state = core.apply_action(state, core.Action(kind="pass", raw="pass")).state
            if normalized == bad_message:
                bad_streak += 1
            else:
                bad_message = normalized
                bad_streak = 1
        else:
            parsed_turns += 1
            result = core.apply_action(state, action)
            state = result.state
            if not result.legal:
                illegal_actions += 1
                if normalized == bad_message:
                    bad_streak += 1
                else:
                    bad_message = normalized
                    bad_streak = 1
            else:
                bad_message = None
                bad_streak = 0

        if bad_streak >= repeat_stop:
            repeated_messages += 1
            no_progress_stopped = True
            state = replace(
                state, stopped=True, stop_reason=f"repeated no-progress response {repeat_stop} times"
            )
            break
        if core.is_done(state):
            break

    format_ok = parsed_turns == assistant_turns if assistant_turns else False
    components = core.summarize_reward(state, format_ok)
    return core.ReplayResult(
        final_state=state,
        assistant_turns=assistant_turns,
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


def outcome_reward(completion, info, **kwargs) -> float:
    return replay(info, completion).reward


def format_reward(completion, info, **kwargs) -> float:
    return replay(info, completion).components["format"] / core.FORMAT_BONUS


def terminal_return_metric(completion, info, **kwargs) -> float:
    return replay(info, completion).terminal_return


def illegal_count(completion, info, **kwargs) -> float:
    return float(replay(info, completion).illegal_actions)


def num_trades(completion, info, **kwargs) -> float:
    return float(replay(info, completion).num_trades)


def first_trade_turn(completion, info, **kwargs) -> float:
    value = replay(info, completion).first_trade_turn
    return -1.0 if value is None else float(value)


def no_progress_stop_metric(completion, info, **kwargs) -> float:
    return 1.0 if replay(info, completion).no_progress_stopped else 0.0


def final_inventory(completion, info, **kwargs) -> float:
    return float(replay(info, completion).final_state.inventory)


def fill_count(completion, info, **kwargs) -> float:
    return float(replay(info, completion).final_state.fills)


def realized_spread(completion, info, **kwargs) -> float:
    return float(replay(info, completion).final_state.realized_spread)


def pickoff_losses(completion, info, **kwargs) -> float:
    return float(replay(info, completion).final_state.pickoff_losses)


def max_debt_metric(completion, info, **kwargs) -> float:
    return float(replay(info, completion).final_state.max_debt)


def interest_paid_metric(completion, info, **kwargs) -> float:
    return float(replay(info, completion).final_state.interest_paid)


def defaulted_metric(completion, info, **kwargs) -> float:
    return 1.0 if replay(info, completion).final_state.defaulted else 0.0


def avg_quote_width(completion, info, **kwargs) -> float:
    """Average posted quote width (ask - bid) across the transcript's quote
    commands. 0.0 if the model never posted a quote."""
    widths = [
        action.ask - action.bid
        for text in assistant_texts(completion)
        if (action := core.parse_action(text)) is not None
        and action.kind == "quote"
        and action.ask > action.bid  # crossed quotes are rejected, not posted
    ]
    if not widths:
        return 0.0
    return sum(widths) / len(widths)


class BazaarEnv(vf.MultiTurnEnv):
    def __init__(self, strict_format: bool = True, repeat_stop: int = 4, **kwargs):
        super().__init__(max_turns=200, **kwargs)
        self.strict_format = strict_format
        self.repeat_stop = repeat_stop

    async def setup_state(self, state):
        info = state["info"]
        task = task_from_info(info)
        market_state = core.initial_state(task)
        state["market_state"] = market_state
        state["parse_warnings"] = 0
        state["bad_message"] = None
        state["bad_streak"] = 0
        info["repeat_stop"] = self.repeat_stop
        return state

    def _message(self, content: str):
        return [vf.UserMessage(content=content)]

    def _track_bad(self, state, normalized: str) -> None:
        if normalized == state["bad_message"]:
            state["bad_streak"] += 1
        else:
            state["bad_message"] = normalized
            state["bad_streak"] = 1

    async def env_response(self, messages, state, **kwargs):
        text = str(messages[-1].get("content") or "") if messages else ""
        market_state: core.MarketState = state["market_state"]
        normalized = " ".join(text.lower().split())

        action = core.parse_action(text)
        if action is None:
            self._track_bad(state, normalized)
            if self.strict_format and state["parse_warnings"] == 0:
                state["parse_warnings"] += 1
                if state["bad_streak"] < self.repeat_stop:
                    return self._message(
                        "format warning: reply with one command like `buy 3`, "
                        "`sell 2`, or `pass`. No turn consumed.\n"
                        f"{core.render_state(market_state)}"
                    )
            result = core.apply_action(market_state, core.Action(kind="pass", raw="pass"))
            reply = "No valid command found. Wasted a turn."
        else:
            result = core.apply_action(market_state, action)
            reply = result.reply
            if result.legal:
                state["bad_message"] = None
                state["bad_streak"] = 0
            else:
                self._track_bad(state, normalized)

        market_state = result.state
        state["market_state"] = market_state

        if state["bad_streak"] >= self.repeat_stop:
            market_state = replace(
                market_state,
                stopped=True,
                stop_reason=f"repeated no-progress response {self.repeat_stop} times",
            )
            state["market_state"] = market_state
            state["final_env_response"] = self._message(
                f"stopped: repeated no-progress response {self.repeat_stop} times\n"
                f"{reply}\n{core.render_state(market_state)}"
            )
            return state["final_env_response"]
        if core.is_done(market_state):
            state["final_env_response"] = self._message(
                f"episode complete\n{reply}\n{core.render_state(market_state)}"
            )
            return state["final_env_response"]
        return self._message(f"{reply}\n{core.render_state(market_state)}")


def task_row(task: core.MarketTask, strategy_hint: bool = False) -> dict[str, Any]:
    first_state = core.initial_state(task)
    rules = RULES.format(horizon=task.horizon)
    if task.config.credit:
        rules += CREDIT_RULES.format(
            rate_pct=round(task.config.interest_rate * 100, 2),
            max_leverage=task.config.max_leverage,
            maintenance_pct=round(task.config.maintenance_ratio * 100, 1),
        )
    if strategy_hint:
        rules += STRATEGY_HINT
    return {
        "prompt": [
            vf.SystemMessage(content=rules).model_dump(),
            vf.UserMessage(
                content=(
                    f"Task seed={task.seed}, tier={task.tier}. Trade for {task.horizon} turns.\n"
                    f"{core.render_state(first_state)}"
                )
            ).model_dump(),
        ],
        "answer": "",
        "info": {
            "seed": task.seed,
            "tier": task.tier,
            "horizon": task.horizon,
            "starting_cash": task.starting_cash,
            "starting_inventory": task.starting_inventory,
            "index_path": list(task.index_path),
            "config": {
                key: getattr(task.config, key)
                for key in core.Tier.__dataclass_fields__
            },
            "quotes": [
                {
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "bid_size": quote.bid_size,
                    "ask_size": quote.ask_size,
                }
                for quote in task.quotes
            ],
        },
    }


def build_dataset(
    n: int,
    seed0: int,
    tier: str,
    overrides: dict[str, Any] | None = None,
    strategy_hint: bool = False,
) -> Dataset:
    rows = []
    for seed in range(seed0, seed0 + n):
        rows.append(
            task_row(
                core.generate(tier=tier, seed=seed, **(overrides or {})),
                strategy_hint=strategy_hint,
            )
        )
    return Dataset.from_list(rows)


def load_environment(
    num_train_examples: int = 200,
    num_eval_examples: int = 40,
    tier: str = "micro",
    strict_format: bool = True,
    repeat_stop: int = 4,
    strategy_hint: bool = False,
    horizon: int | None = None,
    starting_cash: float | None = None,
    starting_inventory: int | None = None,
    start_index: float | None = None,
    vol: float | None = None,
    spread: float | None = None,
    edge_width: float | None = None,
    edge_probability: float | None = None,
    max_quote_qty: int | None = None,
    credit: bool | None = None,
    interest_rate: float | None = None,
    max_leverage: float | None = None,
    maintenance_ratio: float | None = None,
    **kwargs,
) -> vf.Environment:
    overrides = {
        key: value
        for key, value in (
            ("horizon", horizon),
            ("starting_cash", starting_cash),
            ("starting_inventory", starting_inventory),
            ("start_index", start_index),
            ("vol", vol),
            ("spread", spread),
            ("edge_width", edge_width),
            ("edge_probability", edge_probability),
            ("max_quote_qty", max_quote_qty),
            ("credit", credit),
            ("interest_rate", interest_rate),
            ("max_leverage", max_leverage),
            ("maintenance_ratio", maintenance_ratio),
        )
        if value is not None
    }
    train = build_dataset(num_train_examples, 0, tier, overrides, strategy_hint=strategy_hint)
    evald = build_dataset(num_eval_examples, 1_000_000, tier, overrides, strategy_hint=strategy_hint)

    rubric = vf.Rubric()
    rubric.add_reward_func(outcome_reward, weight=1.0)
    rubric.add_reward_func(format_reward, weight=0.0)
    rubric.add_metric(terminal_return_metric)
    rubric.add_metric(illegal_count)
    rubric.add_metric(num_trades)
    rubric.add_metric(first_trade_turn)
    rubric.add_metric(no_progress_stop_metric)
    rubric.add_metric(final_inventory)
    rubric.add_metric(fill_count)
    rubric.add_metric(realized_spread)
    rubric.add_metric(pickoff_losses)
    rubric.add_metric(avg_quote_width)
    rubric.add_metric(max_debt_metric)
    rubric.add_metric(interest_paid_metric)
    rubric.add_metric(defaulted_metric)

    return BazaarEnv(
        dataset=train,
        eval_dataset=evald,
        rubric=rubric,
        strict_format=strict_format,
        repeat_stop=repeat_stop,
        message_type="chat",
    )
