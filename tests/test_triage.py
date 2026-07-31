"""Tests for the decision about what is worth a person's attention."""

from __future__ import annotations

import numpy as np

from conftest import make_trajectory
from fleet_loop.node.triage import Coverage, score, select_within_budget


def test_a_failure_outranks_a_routine_success():
    coverage = Coverage()
    failure = score(make_trajectory("a", success=False), coverage)
    success = score(make_trajectory("b", success=True), coverage)
    assert failure.value > success.value
    assert failure.top_reason() == "failed"


def test_a_near_miss_outranks_a_comfortable_success():
    """Where the policy is about to fail is invisible in a success rate."""
    coverage = Coverage()
    near = score(make_trajectory("a", success=True, steps=100), coverage)
    comfortable = score(make_trajectory("b", success=True, steps=12), coverage)
    assert near.value > comfortable.value
    assert "near_miss" in near.reasons


def test_device_flags_raise_an_episodes_value():
    coverage = Coverage()
    plain = score(make_trajectory("a", success=True, steps=12), coverage)
    flagged = score(
        make_trajectory("b", success=True, steps=12, flags=["observation_ood:7.2sigma"]),
        coverage,
    )
    assert flagged.value > plain.value


def test_novelty_decays_as_a_region_is_shipped():
    """Two hundred more episodes of the same failure teach nothing."""
    coverage = Coverage()
    first = score(make_trajectory("a", success=False, goal=1.2), coverage)
    coverage.record(make_trajectory("a", success=False, goal=1.2))
    for i in range(6):
        coverage.record(make_trajectory(f"r{i}", success=False, goal=1.2))
    later = score(make_trajectory("b", success=False, goal=1.2), coverage)

    assert first.value > later.value
    assert "novel" in first.reasons
    assert "novel" not in later.reasons


def test_a_different_region_is_still_novel():
    coverage = Coverage()
    for i in range(10):
        coverage.record(make_trajectory(f"r{i}", goal=0.0))
    assert "novel" in score(make_trajectory("far", goal=1.4), coverage).reasons


# -- the budget --------------------------------------------------------------


def make_batch(n: int = 40) -> list:
    return [
        make_trajectory(f"e{i}", success=i % 4 != 0, seed=i, goal=(i % 6) * 0.35, steps=40)
        for i in range(n)
    ]


def test_selection_never_exceeds_the_budget():
    batch = make_batch()
    budget = sum(t.nbytes for t in batch) // 5
    selected, stats = select_within_budget(batch, Coverage(), budget)

    assert stats["bytes_selected"] <= budget
    assert 0 < len(selected) < len(batch)


def test_a_tight_budget_prefers_failures():
    batch = make_batch(60)
    budget = sum(t.nbytes for t in batch) // 10
    selected, stats = select_within_budget(batch, Coverage(), budget)

    chosen_failures = sum(not s.trajectory.success for s in selected)
    assert chosen_failures / len(selected) > 0.4
    assert stats["selected_success_rate"] < stats["observed_success_rate"]


def test_ordinary_successes_get_a_reserved_share():
    """The bias that a failure-only pipeline builds into the next model.

    Uploading nothing but failures drifts the training set to contain only hard
    states, the retrained policy is optimised for a distribution the robot is
    not in, and the next round's failures are worse. The reservation is the fix,
    and without it this test fails.
    """
    # All in one goal region and already well covered, so novelty is not what
    # decides anything here — the contest is failures against plain successes.
    batch = [
        make_trajectory(f"e{i}", success=i % 2 == 0, seed=i, steps=40, goal=0.0) for i in range(60)
    ]
    budget = sum(t.nbytes for t in batch) // 6

    def plain_successes(selection):
        return sum(1 for s in selection if s.trajectory.success and "near_miss" not in s.reasons)

    def covered() -> Coverage:
        coverage = Coverage()
        for trajectory in batch:
            coverage.record(trajectory)
        return coverage

    with_quota, stats = select_within_budget(batch, covered(), budget, success_quota=0.25)
    without, _ = select_within_budget(batch, covered(), budget, success_quota=0.0)

    assert stats["quota_bytes_used"] > 0
    assert plain_successes(without) == 0, "failures should crowd out ordinary episodes"
    assert plain_successes(with_quota) > 0, "the reservation did not get any through"


def test_the_quota_is_not_filled_from_the_top_of_the_ranking():
    """Filling it with near-misses would defeat the purpose.

    The highest-scoring successes are the ones that nearly failed. They are not
    ordinary episodes and they do not correct the skew.
    """
    batch = [make_trajectory(f"slow{i}", success=True, steps=110, seed=i) for i in range(20)]
    batch += [make_trajectory(f"quick{i}", success=True, steps=20, seed=100 + i) for i in range(20)]
    budget = sum(t.nbytes for t in batch) // 4

    selected, _ = select_within_budget(batch, Coverage(), budget, success_quota=0.5)
    quick = sum(1 for s in selected if s.trajectory.episode_id.startswith("quick"))
    assert quick > 0, "the reservation was spent entirely on near-misses"


def test_zero_value_episodes_are_not_shipped_just_because_there_is_room():
    """Budget headroom is not a reason to spend it."""
    batch = [make_trajectory(f"e{i}", success=True, steps=15, seed=i) for i in range(10)]
    coverage = Coverage()
    for trajectory in batch:
        coverage.record(trajectory)  # everything already well covered

    huge = sum(t.nbytes for t in batch) * 10
    selected, stats = select_within_budget(batch, coverage, huge, success_quota=0.0)
    assert selected == []
    assert stats["bytes_selected"] == 0


def test_selection_is_reported_with_its_reasons():
    """The hub should be able to audit what the heuristic is actually shipping."""
    batch = make_batch()
    selected, stats = select_within_budget(batch, Coverage(), sum(t.nbytes for t in batch) // 4)
    assert sum(stats["reasons"].values()) == len(selected)
    assert set(stats["reasons"]) <= {"failed", "near_miss", "device_flagged", "novel", "quota"}


def test_selection_is_deterministic_given_the_same_generator():
    batch = make_batch()
    budget = sum(t.nbytes for t in batch) // 4
    a, _ = select_within_budget(batch, Coverage(), budget, rng=np.random.default_rng(3))
    b, _ = select_within_budget(batch, Coverage(), budget, rng=np.random.default_rng(3))
    assert [s.trajectory.episode_id for s in a] == [s.trajectory.episode_id for s in b]
