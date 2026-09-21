import pytest

from bazaar_env.env import (
    TIER_SEED_STRIDE,
    build_dataset,
    build_mixed_dataset,
    parse_tier_mix,
)

MIX = ["credit_micro", "maker_micro", "micro"]


def _tiers(dataset):
    return [row["tier"] for row in dataset["info"]]


def _keys(dataset):
    return [(row["tier"], row["seed"]) for row in dataset["info"]]


def test_parse_tier_mix_accepts_string_and_list():
    assert parse_tier_mix("credit_micro,maker_micro,micro") == MIX
    assert parse_tier_mix(MIX) == MIX
    assert parse_tier_mix(" micro , maker_micro ") == ["micro", "maker_micro"]


def test_parse_tier_mix_rejects_unknown_and_empty():
    with pytest.raises(ValueError, match="unknown tier"):
        parse_tier_mix("micro,not_a_tier")
    with pytest.raises(ValueError, match="empty"):
        parse_tier_mix(" , ")


def test_mixed_dataset_round_robins_tiers():
    dataset = build_mixed_dataset(9, 0, MIX)

    assert _tiers(dataset) == MIX * 3


def test_mixed_dataset_tasks_are_unique():
    dataset = build_mixed_dataset(60, 0, MIX)
    keys = _keys(dataset)

    assert len(set(keys)) == len(keys)


def test_repeated_tier_weights_the_mix_without_duplicating_tasks():
    """Repeating a tier is how a mix gets weighted, so the repeated slots must
    draw different seeds rather than the same task twice."""

    weighted = ["credit_micro", "credit_micro", "micro"]
    dataset = build_mixed_dataset(30, 0, weighted)
    tiers = _tiers(dataset)
    keys = _keys(dataset)

    assert tiers.count("credit_micro") == 20
    assert tiers.count("micro") == 10
    assert len(set(keys)) == len(keys)

    index_paths = [tuple(row["index_path"]) for row in dataset["info"]]
    assert len(set(index_paths)) == len(index_paths)


def test_train_seeds_stay_below_the_frozen_eval_block():
    dataset = build_mixed_dataset(240, 0, MIX)

    assert max(row["seed"] for row in dataset["info"]) < 1_000_000


def test_train_and_eval_mixes_are_seed_disjoint():
    train = build_mixed_dataset(60, 0, MIX)
    evald = build_mixed_dataset(60, 1_000_000, MIX)

    assert not set(_keys(train)) & set(_keys(evald))


def test_each_slot_gets_its_own_seed_block():
    """Identical integer seeds share one RNG stream, so slots must not collide."""

    dataset = build_mixed_dataset(len(MIX), 0, MIX)
    seeds = [row["seed"] for row in dataset["info"]]

    assert seeds == [slot * TIER_SEED_STRIDE for slot in range(len(MIX))]


def test_mixed_rows_carry_their_own_tier_rules():
    dataset = build_mixed_dataset(3, 0, MIX)
    systems = [row["prompt"][0]["content"] for row in dataset]

    credit_rules, maker_rules, taker_rules = systems
    assert "MARGIN CALL" in credit_rules
    assert "MARGIN CALL" not in maker_rules
    assert "MARGIN CALL" not in taker_rules


def test_single_tier_path_is_unchanged():
    """Previous runs' numbers stay comparable only if the single-tier dataset
    is byte-identical to what it was before tier_mix existed."""

    dataset = build_dataset(5, 0, "micro")

    assert _keys(dataset) == [("micro", seed) for seed in range(5)]
