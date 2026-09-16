"""Tests for the contract that crosses the link."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import ACTIONS, OBS_COLUMNS, make_trajectory
from fleet_loop.wire import Trajectory, pack_shard, unpack_shard


def pack(trajectories, **kwargs):
    return pack_shard(trajectories, obs_columns=OBS_COLUMNS, action_columns=ACTIONS, **kwargs)


def test_a_shard_round_trips(trajectories):
    manifest, payload = pack(trajectories)
    restored = unpack_shard(manifest, payload)

    assert len(restored) == len(trajectories)
    assert [t.episode_id for t in restored] == [t.episode_id for t in trajectories]
    assert [t.steps for t in restored] == [t.steps for t in trajectories]
    assert restored[3].success == trajectories[3].success
    # float16 on the wire, so this is a lossy round trip by design — the bound
    # is asserted in units that matter by test_real_trajectories_compress.
    assert np.allclose(restored[0].observations, trajectories[0].observations, atol=3e-3)


def test_a_shard_is_identified_by_its_contents(trajectories):
    """Two nodes that hit the same failure should not each pay to ship it, and a
    node retrying an upload should be recognised rather than duplicated."""
    first, _ = pack(trajectories[:4])
    again, _ = pack(trajectories[:4])
    different, _ = pack(trajectories[1:5])

    assert first.shard_id == again.shard_id
    assert first.shard_id != different.shard_id


def test_a_truncated_payload_is_detected(trajectories):
    manifest, payload = pack(trajectories[:4])
    with pytest.raises(ValueError, match="hashes to"):
        unpack_shard(manifest, payload[:-32])


def test_a_shard_may_not_mix_nodes_or_versions(trajectories):
    """A shard that cannot be attributed is a shard the canary cannot use."""
    mixed_nodes = [
        make_trajectory("a", node_id="node-a"),
        make_trajectory("b", node_id="node-b"),
    ]
    with pytest.raises(ValueError, match="one node running one policy version"):
        pack(mixed_nodes)

    mixed_versions = [
        make_trajectory("a", policy_version=1),
        make_trajectory("b", policy_version=2),
    ]
    with pytest.raises(ValueError, match="one node running one policy version"):
        pack(mixed_versions)


def test_an_empty_shard_is_refused():
    with pytest.raises(ValueError, match="not worth an upload"):
        pack([])


def test_trajectories_must_have_matching_observation_and_action_counts():
    with pytest.raises(ValueError, match="observations but"):
        Trajectory(
            episode_id="e",
            node_id="n",
            policy_version=1,
            observations=np.zeros((10, 18), np.float32),
            actions=np.zeros((9, 8), np.float32),
            success=True,
            final_distance=0.0,
        )


def test_real_trajectories_compress_on_the_wire():
    """The budget is in wire bytes; triage can only see uncompressed ones.

    Measured on actual rollouts rather than on synthetic noise, because the
    ratio is a property of the data and float32 noise does not compress at all.
    This is the number :class:`BandwidthEstimate` exists to track.
    """
    from fleet_loop.sim import ExpertPolicy, ReachEnv, long_reach

    env = ReachEnv(env=long_reach())
    expert = ExpertPolicy()
    episodes = []
    for i in range(12):
        obs = env.reset(seed=i)
        observations, actions = [], []
        while True:
            observations.append(obs)
            obs, done, ok = env.step(expert.forward(obs))
            actions.append(env.commanded)
            if done:
                break
        episodes.append(
            Trajectory(
                episode_id=f"e{i}",
                node_id="node-a",
                policy_version=1,
                observations=np.asarray(observations, np.float32),
                actions=np.asarray(actions, np.float32),
                success=ok,
                final_distance=env.distance,
            )
        )

    manifest, payload = pack(episodes)
    raw = sum(t.nbytes for t in episodes)
    assert manifest.n_bytes == len(payload)
    assert (
        raw / manifest.n_bytes > 1.5
    ), "the wire encoding regressed; the budget maths depends on it"

    # And the precision that buys is bounded, in the units that matter.
    restored = unpack_shard(manifest, payload)
    obs_error = max(
        float(np.abs(a.observations - b.observations).max())
        for a, b in zip(episodes, restored, strict=True)
    )
    action_error = max(
        float(np.abs(a.actions - b.actions).max()) for a, b in zip(episodes, restored, strict=True)
    )
    assert obs_error < 3e-3, "joint angles quantised past the encoder's own resolution"
    assert action_error < 2e-4, "actions quantised into a range the actuator can distinguish"


def test_an_outcome_is_orders_of_magnitude_cheaper_than_a_trajectory():
    trajectory = make_trajectory("e", steps=120)
    assert trajectory.outcome().nbytes() < trajectory.nbytes / 50


def test_an_outcome_carries_everything_the_canary_needs():
    trajectory = make_trajectory("e", node_id="node-c", policy_version=4, success=False)
    outcome = trajectory.outcome()
    assert (outcome.node_id, outcome.policy_version, outcome.success) == ("node-c", 4, False)


def test_manifest_reasons_survive_the_wire(trajectories):
    manifest, _ = pack(trajectories[:4], value=2.5, reasons={"failed": 3, "novel": 1})
    assert manifest.value == 2.5
    assert manifest.reasons == {"failed": 3, "novel": 1}
    assert manifest.n_episodes == 4
