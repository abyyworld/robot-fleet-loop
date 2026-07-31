"""Tests for how episodes become a training set, and whether it trains."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import OBS_COLUMNS, make_trajectory
from fleet_loop.hub.dataset import assemble, coverage_gain
from fleet_loop.hub.train import TrainConfig, train
from fleet_loop.sim import ExpertPolicy, ReachEnv, ZeroPolicy, long_reach, nominal, success_rate


def batch(n=20, **kwargs):
    return [make_trajectory(f"e{i}", seed=i, **kwargs) for i in range(n)]


# -- assembly ----------------------------------------------------------------


def test_a_dataset_is_identified_by_the_shards_in_it():
    """The join key between a checkpoint and the data that produced it."""
    trajectories = batch()
    first = assemble(trajectories, ["s1", "s2"]).version
    same = assemble(trajectories, ["s2", "s1"]).version  # order must not matter
    different = assemble(trajectories, ["s1", "s3"]).version

    assert first.dataset_hash == same.dataset_hash
    assert first.dataset_hash != different.dataset_hash


def test_composition_is_recorded_so_drift_is_visible():
    trajectories = [
        *batch(6, node_id="node-a", policy_version=1),
        *batch(4, node_id="node-b", policy_version=2),
    ]
    version = assemble(trajectories, ["s1"]).version

    assert version.episodes_by_node == {"node-a": 6, "node-b": 4}
    assert version.episodes_by_policy_version == {"1": 6, "2": 4}


def test_failures_are_relabelled_and_successes_are_not():
    """A failed episode's actions did not work. Training on them teaches the
    policy to reproduce the failure."""
    trajectories = [
        make_trajectory("ok", success=True, seed=1),
        make_trajectory("bad", success=False, seed=2),
    ]
    training_set = assemble(trajectories, ["s1"])

    assert training_set.version.relabelled_episodes == 1
    assert training_set.version.demonstrated_episodes == 1

    # The successful episode's actions survive verbatim; the failure's do not.
    success_actions = trajectories[0].actions
    assert any(
        np.allclose(training_set.actions[i : i + len(success_actions)], success_actions)
        for i in (0, len(trajectories[1].actions))
    )


def test_the_correction_magnitude_is_measured():
    """How wrong the deployed policy is in the states it is failing in, in
    radians — the most useful single number the hub produces."""
    version = assemble(batch(10, success=False), ["s1"]).version
    assert version.mean_correction > 0
    assert assemble(batch(10, success=True), ["s1"]).version.mean_correction == 0


def test_relabelling_can_be_turned_off_and_the_record_says_so():
    training_set = assemble(batch(10, success=False), ["s1"], relabel_failures=False)
    assert training_set.version.relabelled_episodes == 0
    assert training_set.version.demonstrated_episodes == 10
    # And the hash differs, because the labelling rule is part of the identity.
    assert (
        training_set.version.dataset_hash
        != assemble(batch(10, success=False), ["s1"]).version.dataset_hash
    )


def test_the_window_keeps_the_most_recent_episodes():
    """The first policy's failures should not outnumber the current one's
    forever."""
    old = [
        make_trajectory(f"old{i}", seed=i, finished_at=f"2026-01-01T00:00:{i:02d}+00:00")
        for i in range(10)
    ]
    new = [
        make_trajectory(f"new{i}", seed=100 + i, finished_at=f"2026-06-01T00:00:{i:02d}+00:00")
        for i in range(10)
    ]
    training_set = assemble(old + new, ["s1"], window=10)

    assert training_set.version.n_episodes == 10
    # The ten kept are the recent ones: their observations are the ones present.
    kept = training_set.observations
    assert any(np.allclose(kept[: len(t.observations)], t.observations) for t in new)
    assert not any(np.allclose(kept[: len(t.observations)], t.observations) for t in old)


def test_coverage_gain_is_about_new_shards_not_new_bytes():
    first = assemble(batch(), ["s1", "s2"]).version
    overlapping = assemble(batch(), ["s1", "s2", "s3"]).version
    disjoint = assemble(batch(), ["s4", "s5"]).version

    assert coverage_gain(first, None) == 1.0
    assert coverage_gain(overlapping, first) == pytest.approx(1 / 3)
    assert coverage_gain(disjoint, first) == 1.0


def test_an_empty_dataset_is_refused():
    with pytest.raises(ValueError, match="no episodes"):
        assemble([], [])


# -- training ----------------------------------------------------------------


def expert_data(environment, episodes=120, seed=0):
    env = ReachEnv(env=environment)
    expert = ExpertPolicy()
    observations, actions = [], []
    for i in range(episodes):
        obs = env.reset(seed=seed * 1000 + i)
        while True:
            observations.append(obs)
            obs, done, _ = env.step(expert.forward(obs))
            actions.append(env.commanded)
            if done:
                break
    return np.asarray(observations, np.float32), np.asarray(actions, np.float32)


def test_training_produces_a_policy_that_actually_works():
    """A loop whose training step does not train is not a loop."""
    observations, actions = expert_data(nominal(), episodes=250)
    policy, report = train(observations, actions, OBS_COLUMNS, config=TrainConfig(seed=0))

    assert report.val_joint_mae_rad < 0.03
    assert success_rate(policy, nominal(), episodes=40) > 0.7


def test_normalisation_is_folded_in_so_the_artifact_is_a_plain_mlp():
    """A normalisation constant that lives outside the model is one that will
    eventually be applied twice, or not at all."""
    observations, actions = expert_data(nominal(), episodes=60)
    policy, _ = train(observations, actions, OBS_COLUMNS, config=TrainConfig(epochs=20))

    # Nothing but weights and biases: the device runs Policy.forward as-is.
    assert len(policy.weights) == len(policy.biases) == 3
    assert policy.forward(observations[:4]).shape == (4, 8)
    assert policy.obs_columns == OBS_COLUMNS


def test_a_policy_trained_on_one_environment_does_not_generalise():
    """The premise of the whole repo, asserted rather than assumed.

    If a policy trained on the nominal node were already good everywhere, there
    would be nothing for a fleet loop to collect.
    """
    observations, actions = expert_data(nominal(), episodes=250)
    policy, _ = train(observations, actions, OBS_COLUMNS, config=TrainConfig(seed=0))

    assert success_rate(policy, nominal(), 40) > 0.7
    assert success_rate(policy, long_reach(), 40) < 0.2


def test_training_on_fleet_data_closes_the_gap():
    """And the fix, asserted too: the same architecture, trained on data from
    the node it was failing on, handles it."""
    nominal_obs, nominal_act = expert_data(nominal(), episodes=150)
    far_obs, far_act = expert_data(long_reach(), episodes=150, seed=2)
    policy, _ = train(
        np.concatenate([nominal_obs, far_obs]),
        np.concatenate([nominal_act, far_act]),
        OBS_COLUMNS,
        config=TrainConfig(seed=0),
    )
    assert success_rate(policy, long_reach(), 40) > 0.7


def test_too_few_samples_is_refused():
    with pytest.raises(ValueError, match="not enough to train"):
        train(np.zeros((50, 18), np.float32), np.zeros((50, 8), np.float32), OBS_COLUMNS)


def test_mismatched_observations_and_actions_are_refused():
    with pytest.raises(ValueError, match="observations but"):
        train(np.zeros((200, 18), np.float32), np.zeros((150, 8), np.float32), OBS_COLUMNS)


# -- the task itself ---------------------------------------------------------


@pytest.mark.parametrize("environment", [nominal(), long_reach()])
def test_the_task_discriminates(environment):
    """A suite on which a do-nothing policy scores well is measuring the
    environment. Both anchors, on every node's conditions."""
    assert success_rate(ZeroPolicy(), environment, episodes=40) == 0.0
    assert success_rate(ExpertPolicy(), environment, episodes=40) > 0.95


def test_success_requires_holding_the_goal_not_touching_it():
    """A policy that flails through the tolerance ball has not reached the goal.

    This is the mistake the evaluation harness upstream found the hard way; a
    latching success criterion ranks a random policy well above zero.
    """
    env = ReachEnv(env=nominal())
    env.reset(seed=0)
    env.joints = env.goal.copy()

    _, done, success = env.step(np.zeros(8, np.float32))
    assert not success, "success latched on the first step inside tolerance"

    for _ in range(4):
        _, done, success = env.step(np.zeros(8, np.float32))
    assert success, "holding the goal for settle_steps should succeed"
