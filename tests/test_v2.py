from bazaar_env import core
from bazaar_env.env import replay, task_row


def test_maker_tier_starts_without_standing_quote():
    state = core.initial_state(core.generate(tier="maker_micro", seed=0))

    assert state.standing_quote is None
    assert core.public_state(state)["standing_quote"] is None


def test_quote_action_posts_two_sided_market_and_flow_fills():
    task = core.generate(tier="maker_micro", seed=0)
    state = core.initial_state(task)
    action = core.vol_aware_quoter_policy(state)

    result = core.apply_action(state, action)

    assert result.legal
    assert result.state.turn == 1
    assert result.state.standing_quote is not None
    assert result.state.fills >= 0
    assert result.state.realized_spread >= 0


def test_self_crossed_quote_is_illegal():
    state = core.initial_state(core.generate(tier="maker_micro", seed=0))
    action = core.Action(kind="quote", bid=101, bid_size=1, ask=100, ask_size=1, raw="quote 101 1 100 1")

    result = core.apply_action(state, action)

    assert not result.legal
    assert "bid must be below ask" in result.reply


def test_pickoff_accounting_on_stale_quote():
    state = core.initial_state(core.generate(tier="maker_micro", seed=0))
    # Too tight and stale: guarantees informed flow can pick it off after the
    # index step on at least some seeded turns.
    action = core.Action(kind="quote", bid=100.0, bid_size=5, ask=100.1, ask_size=5, raw="quote 100 5 100.1 5")
    result = core.apply_action(state, action)

    assert result.state.fills > 0
    assert result.state.pickoff_losses >= 0
    assert any(trade.source in {"noise", "pickoff"} for trade in result.state.trades)


def test_wide_quoter_gets_no_fills_on_seed_zero():
    result = core.run_policy(core.generate(tier="maker_micro", seed=0), core.wide_quoter_policy)

    assert result.final_state.fills == 0
    assert result.num_trades == 0


def test_tight_quoter_gets_more_pickoff_than_vol_aware_average():
    tight_losses = []
    aware_losses = []
    for seed in range(10):
        task = core.generate(tier="maker_micro", seed=seed)
        tight_losses.append(core.run_policy(task, core.tight_quoter_policy).final_state.pickoff_losses)
        aware_losses.append(core.run_policy(task, core.vol_aware_quoter_policy).final_state.pickoff_losses)

    assert sum(tight_losses) > sum(aware_losses)


def test_vol_aware_quoter_beats_wide_quoter_average():
    wide = []
    aware = []
    for seed in range(20):
        task = core.generate(tier="maker_micro", seed=seed)
        wide.append(core.run_policy(task, core.wide_quoter_policy).terminal_return)
        aware.append(core.run_policy(task, core.vol_aware_quoter_policy).terminal_return)

    assert sum(aware) / len(aware) > sum(wide) / len(wide)


def test_bait_depth_can_reduce_executable_size():
    task = core.generate(tier="maker_micro", seed=0, bait_probability=1.0, bait_depth_fraction=0.25)
    state = core.initial_state(task)
    quote = core.current_quote(state)
    idx = core.current_index(state)
    # Make sure the quote is attractive enough to trigger baiting.
    baited = core.replace(task, quotes=(core.Quote(bid=quote.bid, ask=idx - 3, bid_size=quote.bid_size, ask_size=8),) + task.quotes[1:])
    state = core.initial_state(baited)
    action = core.Action(kind="buy", quantity=8, raw="buy 8")

    result = core.apply_action(state, action)

    assert not result.legal
    assert "displayed ask size" in result.reply


def test_trigger_hunt_moves_quote_inside_prior_threshold():
    task = core.generate(tier="maker_micro", seed=1, trigger_hunt_probability=1.0)
    state = core.initial_state(task)
    # Create a prior profitable buy edge of 2.0.
    trade = core.Trade(turn=1, side="buy", quantity=1, price=98, index_value=100, source="take", edge_vs_index=2.0)
    state = core.replace(state, trades=(trade,))

    quote = core.current_quote(state)

    assert quote.ask > 98
    assert quote.ask < core.current_index(state)


def test_replay_parses_quote_command_and_records_fills():
    task = core.generate(tier="maker_micro", seed=0)
    completion = [{"role": "assistant", "content": "quote 99 3 101 3"}]

    result = replay(task_row(task)["info"], completion)

    assert result.parsed_turns == 1
    assert result.final_state.standing_quote is not None
    assert result.final_state.turn == 1


def test_result_dict_includes_maker_metrics():
    result = core.run_policy(core.generate(tier="maker_micro", seed=0), core.vol_aware_quoter_policy)
    row = core.result_to_dict(result)

    assert {"fills", "realized_spread", "pickoff_losses"} <= set(row)


# --- Regression tests from the 09-18 v2 review ---


def test_taker_trades_record_edge_and_source():
    # Bug: taker trades defaulted edge_vs_index to 0.0, which silently killed
    # trigger-hunting (it needs a positive prior take edge).
    task = core.generate(tier="micro", seed=2)
    state = core.initial_state(task)
    quote = core.current_quote(state)
    index = core.current_index(state)

    result = core.apply_action(state, core.Action(kind="buy", quantity=1, raw="buy 1"))

    assert result.trade is not None
    assert result.trade.source == "take"
    assert result.trade.edge_vs_index == core.round_money(index - quote.ask)


def test_trigger_hunt_fires_after_real_take():
    # End-to-end: a real profitable take through apply_action must arm the hunt.
    for seed in range(30):
        task = core.generate(tier="maker_micro", seed=seed, trigger_hunt_probability=1.0)
        state = core.initial_state(task)
        quote = core.current_quote(state)
        index = core.current_index(state)
        if quote.ask < index and state.cash >= quote.ask:
            state = core.apply_action(state, core.Action(kind="buy", quantity=1, raw="buy 1")).state
            hunted = core.current_quote(state)
            next_index = core.current_index(state)
            # With hunt probability 1.0 and an armed threshold, the displayed
            # ask sits just inside the revealed edge (below index).
            assert hunted.ask < next_index
            return
    raise AssertionError("no seed produced a profitable turn-0 take")


def test_per_turn_fills_capped_by_posted_size():
    # Bug: pickoff looped intensity times at full size and noise ignored size,
    # so displayed size did not bound per-turn exposure.
    task = core.generate(tier="maker_micro", seed=3)
    state = core.initial_state(task)
    index = core.current_index(state)
    stale_ask = core.round_money(index - 6)
    action = core.Action(
        kind="quote", bid=core.round_money(stale_ask - 1), bid_size=1,
        ask=stale_ask, ask_size=3, raw=f"quote {stale_ask - 1} 1 {stale_ask} 3",
    )

    result = core.apply_action(state, action)

    sell_fills = sum(t.quantity for t in result.state.trades if t.side == "sell")
    buy_fills = sum(t.quantity for t in result.state.trades if t.side == "buy")
    assert sell_fills <= 3
    assert buy_fills <= 1


def test_maker_fill_turn_matches_game_turn():
    # Bug: fills recorded turn+2 because the turn advanced before _fill.
    for seed in range(20):
        task = core.generate(tier="maker_micro", seed=seed)
        state = core.initial_state(task)
        result = core.apply_action(state, core.tight_quoter_policy(state))
        if result.state.trades:
            assert result.state.trades[0].turn == 1
            return
    raise AssertionError("no seed produced a turn-1 fill for a tight quote")


def test_v1_public_state_has_no_maker_keys():
    # Bug: maker keys leaked into v1 state JSON, changing the prompt format the
    # v1 training run was evaluated on.
    v1_state = core.public_state(core.initial_state(core.generate(tier="micro", seed=0)))
    v2_state = core.public_state(core.initial_state(core.generate(tier="maker_micro", seed=0)))

    assert "standing_quote" not in v1_state
    assert "fills" not in v1_state
    assert "standing_quote" in v2_state
    assert "fills" in v2_state


def test_avg_quote_width_measures_posted_widths():
    from bazaar_env.env import avg_quote_width

    completion = [
        {"role": "assistant", "content": "quote 98 3 102 3"},
        {"role": "assistant", "content": "pass"},
        {"role": "assistant", "content": "quote 99 3 101 3"},
    ]
    info = _task_info_maker()

    assert avg_quote_width(completion, info) == 3.0


def _task_info_maker():
    return task_row(core.generate(tier="maker_micro", seed=0))["info"]
