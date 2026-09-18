from bazaar_env import core


def test_generate_is_deterministic_by_seed():
    assert core.generate(seed=7) == core.generate(seed=7)
    assert core.generate(seed=7) != core.generate(seed=8)


def test_micro_tier_has_ten_turns_and_visible_start_state():
    task = core.generate(tier="micro", seed=1)
    state = core.initial_state(task)
    public = core.public_state(state)

    assert task.horizon == 10
    assert public["turns_remaining"] == 10
    assert public["cash"] == 1000.0
    assert public["inventory"] == 5  # dealer book: sell edges live from turn 1
    assert public["mark_to_index_pnl"] == 0.0
    assert "index_value" in public


def test_buy_executes_at_ask_and_advances_turn():
    task = core.generate(seed=2)
    state = core.initial_state(task)
    quote = core.current_quote(state)

    result = core.apply_action(state, core.Action(kind="buy", quantity=1, raw="buy 1"))

    assert result.legal
    assert result.trade is not None
    assert result.trade.price == quote.ask
    assert result.state.inventory == task.starting_inventory + 1
    assert result.state.cash == core.round_money(task.starting_cash - quote.ask)
    assert result.state.turn == 1


def test_sell_requires_inventory():
    # trivial starts with an empty book, so any sell is illegal.
    task = core.generate(tier="trivial", seed=2)
    state = core.initial_state(task)

    result = core.apply_action(state, core.Action(kind="sell", quantity=1, raw="sell 1"))

    assert not result.legal
    assert "not enough inventory" in result.reply or "exceeds bid size" in result.reply
    assert result.state.turn == 0
    assert result.state.illegal_free_used


def test_second_illegal_wastes_turn():
    task = core.generate(tier="trivial", seed=2)
    state = core.initial_state(task)
    action = core.Action(kind="sell", quantity=1, raw="sell 1")

    first = core.apply_action(state, action)
    second = core.apply_action(first.state, action)

    assert not first.legal
    assert not second.legal
    assert first.state.turn == 0
    assert second.state.turn == 1


def test_buy_is_capped_by_ask_size_and_cash():
    task = core.generate(seed=3)
    state = core.initial_state(task)
    quote = core.current_quote(state)

    too_large = core.apply_action(
        state, core.Action(kind="buy", quantity=quote.ask_size + 1, raw=f"buy {quote.ask_size + 1}")
    )
    too_expensive = core.apply_action(
        state, core.Action(kind="buy", quantity=10_000, raw="buy 10000")
    )

    assert "exceeds ask size" in too_large.reply
    assert "exceeds ask size" in too_expensive.reply or "not enough cash" in too_expensive.reply


def test_sell_is_capped_by_bid_size():
    task = core.generate(seed=4)
    state = core.initial_state(task)
    state = core.apply_action(state, core.Action(kind="buy", quantity=5, raw="buy 5")).state
    quote = core.current_quote(state)

    result = core.apply_action(
        state, core.Action(kind="sell", quantity=quote.bid_size + 1, raw=f"sell {quote.bid_size + 1}")
    )

    assert not result.legal
    assert "exceeds bid size" in result.reply or "not enough inventory" in result.reply


def test_marked_net_worth_uses_current_index_not_trade_price():
    task = core.generate(seed=5)
    state = core.initial_state(task)
    result = core.apply_action(state, core.Action(kind="buy", quantity=1, raw="buy 1"))
    expected = core.round_money(
        result.state.cash + result.state.inventory * core.current_index(result.state)
    )

    assert core.marked_net_worth(result.state) == expected


def test_reward_components_first_trade_bonus_once_only():
    task = core.generate(seed=6)
    state = core.initial_state(task)
    one = core.apply_action(state, core.Action(kind="buy", quantity=1, raw="buy 1")).state
    two = core.apply_action(one, core.Action(kind="buy", quantity=1, raw="buy 1")).state

    assert core.summarize_reward(one, format_ok=True)["first_trade_bonus"] == core.FIRST_TRADE_BONUS
    assert core.summarize_reward(two, format_ok=True)["first_trade_bonus"] == core.FIRST_TRADE_BONUS


def test_passive_policy_runs_full_horizon_without_stop():
    # Passing is a legal action, so repeated passes are progress, not a loop.
    task = core.generate(seed=7)
    result = core.run_policy(task, core.passive_policy)

    # Passive holds the starting book, so its return marks that inventory at
    # the final index — near 1.0, drifting with the index.
    expected = core.round_money(
        task.starting_cash + task.starting_inventory * task.index_path[-1]
    ) / core.starting_net_worth(task)

    assert not result.no_progress_stopped
    assert result.assistant_turns == task.horizon
    assert result.final_state.turn == task.horizon
    assert result.num_trades == 0
    assert abs(result.terminal_return - expected) < 1e-9


def test_loop_spammer_is_stopped_and_does_not_trade():
    task = core.generate(seed=8)
    result = core.run_policy(task, core.loop_spammer_policy)

    assert result.no_progress_stopped
    assert result.illegal_actions >= 2
    assert result.num_trades == 0


def test_honest_dealer_beats_passive_on_micro_average():
    honest = []
    passive = []
    for seed in range(20):
        task = core.generate(tier="micro", seed=seed)
        honest.append(core.run_policy(task, core.honest_dealer_policy).terminal_return)
        passive.append(core.run_policy(task, core.passive_policy).terminal_return)

    assert sum(honest) / len(honest) > sum(passive) / len(passive)


def test_parse_action_extracts_last_command_from_free_text():
    # Lenient last-match extraction is the proven Magic Sort protocol.
    assert core.parse_action("buy 3") == core.Action(kind="buy", quantity=3, raw="buy 3")
    assert core.parse_action("SELL 2") == core.Action(kind="sell", quantity=2, raw="sell 2")
    assert core.parse_action(" pass ") == core.Action(kind="pass", raw="pass")
    assert core.parse_action("The ask is cheap, so I buy 3") == core.Action(
        kind="buy", quantity=3, raw="buy 3"
    )
    assert core.parse_action("I could pass, but instead: buy 2") == core.Action(
        kind="buy", quantity=2, raw="buy 2"
    )
    assert core.parse_action("I think I should buy.") is None
    assert core.parse_action("hold") is None


def test_result_to_dict_contains_exploit_fields():
    result = core.run_policy(core.generate(seed=9), core.honest_dealer_policy)
    row = core.result_to_dict(result)

    assert {"reward", "terminal_return", "num_trades", "final_net_worth", "components"} <= set(row)
