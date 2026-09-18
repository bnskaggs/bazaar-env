import asyncio

import verifiers as vf

from bazaar_env import core
from bazaar_env.env import BazaarEnv, build_dataset, replay, task_row


def _info(seed: int = 0, tier: str = "micro"):
    return task_row(core.generate(tier=tier, seed=seed))["info"]


def test_replay_scores_server_authoritative_transcript():
    task = core.generate(seed=0)
    quote = task.quotes[0]
    completion = [{"role": "assistant", "content": "buy 1"}]

    result = replay(task_row(task)["info"], completion)

    assert result.num_trades == 1
    assert result.final_state.inventory == task.starting_inventory + 1
    assert result.final_state.cash == core.round_money(task.starting_cash - quote.ask)
    assert result.components["first_trade_bonus"] == core.FIRST_TRADE_BONUS


def test_replay_first_format_warning_is_free():
    completion = [
        {"role": "assistant", "content": "I think I should buy."},
        {"role": "assistant", "content": "buy 1"},
    ]

    result = replay(_info(1), completion)

    assert result.assistant_turns == 2
    assert result.parsed_turns == 1
    assert result.final_state.turn == 1
    assert result.components["format"] == 0.0


def test_replay_stops_consecutive_repeated_illegal_messages():
    completion = [{"role": "assistant", "content": "sell 999"} for _ in range(4)]

    result = replay(_info(2), completion)

    assert result.no_progress_stopped
    assert result.repeated_messages == 1


def test_replay_buy_and_hold_is_not_stopped_and_marks_at_final_index():
    # Repeated legal passes are a legitimate strategy, not a no-progress loop.
    info = _info(5)
    completion = [{"role": "assistant", "content": "buy 1"}] + [
        {"role": "assistant", "content": "pass"} for _ in range(9)
    ]

    result = replay(info, completion)

    assert not result.no_progress_stopped
    assert result.num_trades == 1
    assert result.final_state.turn == info["horizon"]
    expected = core.round_money(
        result.final_state.cash + result.final_state.inventory * info["index_path"][-1]
    )
    assert core.marked_net_worth(result.final_state) == expected


def test_env_response_returns_typed_messages_and_state_json():
    env = BazaarEnv(dataset=build_dataset(1, 0, "micro"))
    state = asyncio.run(env.setup_state({"info": _info(3)}))

    messages = asyncio.run(env.env_response([vf.AssistantMessage(content="buy 1")], state))

    assert isinstance(messages[0], vf.UserMessage)
    assert "ok: buy" in messages[0].content
    assert '"cash"' in messages[0].content
    assert state["market_state"].turn == 1


def test_env_response_explains_illegal_and_first_one_is_free():
    # trivial has an empty starting book, so selling is illegal.
    env = BazaarEnv(dataset=build_dataset(1, 0, "trivial"))
    state = asyncio.run(env.setup_state({"info": _info(4, tier="trivial")}))

    first = asyncio.run(env.env_response([vf.AssistantMessage(content="sell 1")], state))
    second = asyncio.run(env.env_response([vf.AssistantMessage(content="sell 1")], state))

    assert "not enough inventory" in first[0].content or "exceeds bid size" in first[0].content
    assert "No turn consumed" in first[0].content
    assert state["market_state"].turn == 1
    assert "Wasted a turn" in second[0].content


def test_build_dataset_uses_disjoint_seed_ranges():
    train = build_dataset(3, 0, "micro")
    evald = build_dataset(3, 1_000_000, "micro")

    train_seeds = {row["info"]["seed"] for row in train}
    eval_seeds = {row["info"]["seed"] for row in evald}

    assert train_seeds == {0, 1, 2}
    assert eval_seeds == {1_000_000, 1_000_001, 1_000_002}
    assert train_seeds.isdisjoint(eval_seeds)


def test_load_environment_exposes_datasets_and_rubric():
    from bazaar_env import load_environment

    env = load_environment(num_train_examples=2, num_eval_examples=2)

    assert env.dataset is not None
    assert env.eval_dataset is not None
    assert env.rubric is not None
