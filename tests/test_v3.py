from dataclasses import replace

from bazaar_env import core
from bazaar_env.env import replay, task_row


def _credit_task(seed: int = 0, **overrides):
    return core.generate(tier="credit_micro", seed=seed, **overrides)


def _find_cheap_ask(seeds=range(50), min_cost_over_cash: float = 1.0):
    """Find a seed whose turn-0 ask is takeable at full size only with margin."""
    for seed in seeds:
        task = _credit_task(seed)
        state = core.initial_state(task)
        quote = core.current_quote(state)
        cost = quote.ask * quote.ask_size
        if quote.ask < core.current_index(state) and cost > state.cash * min_cost_over_cash:
            return task, state, quote
    raise AssertionError("no seed with a margin-requiring cheap ask found")


def test_auto_margin_draw_on_take():
    task, state, quote = _find_cheap_ask()

    result = core.apply_action(
        state, core.Action(kind="buy", quantity=quote.ask_size, raw=f"buy {quote.ask_size}")
    )

    assert result.legal
    assert result.state.debt > 0 or result.state.interest_paid > 0
    assert "margin draw" in result.reply


def test_buying_power_cap_is_named():
    task = _credit_task(0)
    state = core.initial_state(task)

    result = core.apply_action(state, core.Action(kind="buy", quantity=9_999, raw="buy 9999"))

    assert not result.legal
    assert "exceeds ask size" in result.reply or "buying power" in result.reply


def test_auto_repay_from_surplus_cash():
    task = _credit_task(0)
    state = core.initial_state(task)
    state = replace(state, debt=100.0, cash=60.0)

    settled, notes = core.settle_turn(state)

    assert settled.debt < 100.0
    assert settled.cash == 0.0
    assert any("auto-repaid" in note for note in notes)


def test_interest_accrues_on_outstanding_debt():
    task = _credit_task(0)
    state = core.initial_state(task)
    state = replace(state, debt=500.0, cash=0.0)

    settled, notes = core.settle_turn(state)

    assert settled.debt > 500.0
    assert settled.interest_paid > 0
    assert any("interest" in note for note in notes)


def test_margin_call_then_default():
    task = _credit_task(0)
    state = core.initial_state(task)
    # Deep breach: equity far below 30% of debt.
    state = replace(state, debt=1_000.0, cash=0.0, inventory=1)

    called, notes1 = core.settle_turn(state)
    assert called.margin_call
    assert any("MARGIN CALL" in note for note in notes1)
    assert not called.defaulted

    defaulted, notes2 = core.settle_turn(called)
    assert defaulted.defaulted
    assert defaulted.stopped
    assert any("DEFAULT" in note for note in notes2)


def test_default_zeroes_terminal_component():
    task = _credit_task(0)
    state = core.initial_state(task)
    state = replace(state, defaulted=True, stopped=True)

    components = core.summarize_reward(state, format_ok=True)

    assert components["terminal_return"] == 0.0


def test_margin_call_cure_clears_flag():
    task = _credit_task(0)
    state = core.initial_state(task)
    state = replace(state, debt=1_000.0, cash=0.0, inventory=1, margin_call=True)
    # Cure: enough cash that auto-repay extinguishes the debt entirely
    # (repaying only most of it would leave equity ~0 — still in breach).
    state = replace(state, cash=1_000.0)

    settled, notes = core.settle_turn(state)

    assert not settled.defaulted
    assert not settled.margin_call


def test_borrow_verb_gets_instructive_illegal():
    credit_state = core.initial_state(_credit_task(0))
    plain_state = core.initial_state(core.generate(tier="micro", seed=0))

    credit_result = core.apply_action(
        credit_state, core.Action(kind="borrow", quantity=100, raw="borrow 100")
    )
    plain_result = core.apply_action(
        plain_state, core.Action(kind="borrow", quantity=100, raw="borrow 100")
    )

    assert not credit_result.legal
    assert "automatic" in credit_result.reply
    assert not plain_result.legal
    assert "no credit" in plain_result.reply


def test_equity_subtracts_debt():
    state = core.initial_state(_credit_task(0))
    state = replace(state, debt=200.0)

    assert core.equity(state) == core.round_money(core.marked_net_worth(state) - 200.0)


def test_credit_keys_only_on_credit_tiers():
    credit_state = core.public_state(core.initial_state(_credit_task(0)))
    maker_state = core.public_state(core.initial_state(core.generate(tier="maker_micro", seed=0)))

    assert {"debt", "equity", "borrowable", "margin_call"} <= set(credit_state)
    assert "debt" not in maker_state


def test_margin_taker_beats_unlevered_taker_average():
    margin, unlevered = [], []
    for seed in range(20):
        task = _credit_task(seed)
        margin.append(core.run_policy(task, core.margin_taker_policy(1.0)).terminal_return)
        unlevered.append(core.run_policy(task, core.threshold_taker_policy(1.0)).terminal_return)

    assert sum(margin) / len(margin) > sum(unlevered) / len(unlevered)


def test_margin_gambler_loses_to_passive_average():
    gambler, passive = [], []
    for seed in range(20):
        task = _credit_task(seed)
        gambler.append(core.run_policy(task, core.margin_gambler_policy(seed)).terminal_return)
        passive.append(core.run_policy(task, core.passive_policy).terminal_return)

    assert sum(gambler) / len(gambler) < sum(passive) / len(passive)


def test_replay_handles_borrow_attempt_and_continues():
    task = _credit_task(3)
    completion = [
        {"role": "assistant", "content": "borrow 500"},
        {"role": "assistant", "content": "pass"},
    ]

    result = replay(task_row(task)["info"], completion)

    assert result.illegal_actions >= 1
    assert not result.no_progress_stopped


# --- Regressions from the 09-19 outside review ---


def test_credit_micro_is_taker_only():
    # With maker on, noise reservations scaled with edge_width and a static
    # 12-wide quote earned 1.054 risk-free - the tier's optimal policy ignored
    # credit entirely. The credit tier is taker-only until the noise model is
    # retuned for an integrated tier.
    state = core.initial_state(_credit_task(0))

    result = core.apply_action(
        state,
        core.Action(kind="quote", bid=95, bid_size=3, ask=105, ask_size=3, raw="quote 95 3 105 3"),
    )

    assert not result.legal
    assert "maker tiers" in result.reply
    assert "standing_quote" not in core.public_state(state)
    assert "debt" in core.public_state(state)


def test_terminal_return_floored_at_zero_without_default():
    # Surviving with negative equity must never score worse than defaulting.
    state = core.initial_state(_credit_task(0))
    state = replace(state, debt=2_000.0, cash=0.0, inventory=1)

    components = core.summarize_reward(state, format_ok=True)

    assert core.equity(state) < 0
    assert components["terminal_return"] == 0.0


def test_credit_rules_text_present_on_credit_tier():
    row = task_row(_credit_task(0))
    system_text = row["prompt"][0]["content"]

    assert "margin loan" in system_text
    assert "DEFAULT" in system_text

    plain_row = task_row(core.generate(tier="maker_micro", seed=0))
    assert "margin loan" not in plain_row["prompt"][0]["content"]
