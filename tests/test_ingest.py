"""Tests for what the hub does before it believes a node.

Data from a fleet is the least trustworthy input in the system, and it is used
to train the policy that goes back down to every node.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from conftest import ACTIONS, OBS_COLUMNS, make_outcomes, make_trajectory
from fleet_loop.hub.ingest import Ingest, Validator
from fleet_loop.wire import Trajectory, pack_shard


def pack(trajectories, **kwargs):
    return pack_shard(trajectories, obs_columns=OBS_COLUMNS, action_columns=ACTIONS, **kwargs)


@pytest.fixture
def ingest():
    return Ingest(Validator(OBS_COLUMNS, ACTIONS, known_versions={1, 2}))


def test_a_clean_shard_is_accepted(ingest, trajectories):
    manifest, payload = pack(trajectories[:6])
    result = ingest.accept(manifest, payload)

    assert result.ok
    assert len(result.accepted) == 6
    assert ingest.stats()["episodes"] == 6
    assert len(ingest.quarantine) == 0


def test_the_same_shard_twice_is_a_no_op(ingest, trajectories):
    manifest, payload = pack(trajectories[:6])
    ingest.accept(manifest, payload)
    again = ingest.accept(manifest, payload)

    assert again.duplicate
    assert ingest.stats()["episodes"] == 6


def test_offer_lets_a_node_skip_a_payload_it_already_sent(ingest, trajectories):
    manifest, payload = pack(trajectories[:6])
    ingest.accept(manifest, payload)

    accept, already = ingest.offer([manifest])
    assert accept == []
    assert already == [manifest.shard_id]


def test_a_shard_from_an_unpublished_version_cannot_be_attributed(ingest):
    """Not necessarily bad data — but an episode that cannot be tied to a
    release is an episode that cannot judge one, or explain one later."""
    manifest, payload = pack([make_trajectory("e", policy_version=99)])
    result = ingest.accept(manifest, payload)

    assert not result.ok
    assert "provenance" in result.rejections[0].reason
    assert ingest.quarantine.by_reason() == {"provenance": 1}


def test_a_contract_mismatch_is_caught_before_the_payload(ingest, trajectories):
    manifest, payload = pack_shard(
        trajectories[:4],
        obs_columns=[*OBS_COLUMNS, "wrist_force_z"],
        action_columns=ACTIONS,
    )
    accept, _ = ingest.offer([manifest])

    assert accept == [], "a payload was requested for a shard that was always going to fail"
    assert "contract" in ingest.quarantine.items[0].reason
    assert "wrist_force_z" in ingest.quarantine.items[0].reason


def test_a_corrupted_payload_is_quarantined_not_accepted(ingest, trajectories):
    manifest, payload = pack(trajectories[:6])
    result = ingest.accept(manifest, payload[:-64])

    assert not result.ok
    assert "payload" in result.rejections[0].reason
    assert ingest.stats()["episodes"] == 0


def test_non_finite_values_never_reach_the_training_set(ingest):
    """One corrupted shard in the training set is a fleet-wide regression with
    no obvious cause."""
    trajectory = make_trajectory("e")
    trajectory.observations[3, 2] = np.nan
    manifest, payload = pack([trajectory])

    result = ingest.accept(manifest, payload)
    assert not result.ok
    assert "non-finite" in result.rejections[0].reason


def test_an_impossible_action_is_refused(ingest):
    """Beyond what the actuator accepts means the log is not what it claims."""
    trajectory = make_trajectory("e")
    trajectory.actions[5, 1] = 3.0
    manifest, payload = pack([trajectory])

    result = ingest.accept(manifest, payload)
    assert not result.ok
    assert "rad" in result.rejections[0].reason


def test_a_clock_from_the_future_is_refused(ingest, trajectories):
    manifest, payload = pack(trajectories[:4])
    manifest.created_at = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()

    result = ingest.accept(manifest, payload)
    assert not result.ok
    assert "clock" in result.rejections[0].reason


def test_a_naive_timestamp_is_refused(ingest, trajectories):
    manifest, payload = pack(trajectories[:4])
    manifest.created_at = "2026-05-01T10:00:00"

    result = ingest.accept(manifest, payload)
    assert "no timezone" in result.rejections[0].reason


def test_modest_clock_drift_is_tolerated(ingest, trajectories):
    """Clock drift on embedded hardware is normal, not an incident."""
    manifest, payload = pack(trajectories[:4])
    manifest.created_at = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
    assert ingest.accept(manifest, payload).ok


def test_a_shard_from_the_future_schema_is_refused(ingest, trajectories):
    manifest, payload = pack(trajectories[:4])
    manifest.schema_version = 99

    result = ingest.accept(manifest, payload)
    assert "schema" in result.rejections[0].reason


def test_rejections_are_kept_with_a_reason_and_a_node(ingest):
    """A drop is invisible; a quarantine can be looked at.

    The most useful thing a validation layer produces is the list of what was
    dirty and why, because that list is how you find the node with the failing
    sensor.
    """
    for i in range(3):
        trajectory = make_trajectory(f"e{i}", node_id="node-bad")
        trajectory.observations[0, 0] = np.inf
        manifest, payload = pack([trajectory])
        ingest.accept(manifest, payload)

    assert ingest.quarantine.by_node() == {"node-bad": 3}
    assert ingest.quarantine.by_reason() == {"values": 3}


def test_an_episode_with_no_steps_is_refused(ingest):
    trajectory = Trajectory(
        episode_id="e",
        node_id="n",
        policy_version=1,
        observations=np.zeros((0, len(OBS_COLUMNS)), np.float32),
        actions=np.zeros((0, len(ACTIONS)), np.float32),
        success=False,
        final_distance=1.0,
    )
    manifest, payload = pack([trajectory])
    assert "no steps" in ingest.accept(manifest, payload).rejections[0].reason


# -- the cheap channel -------------------------------------------------------


def test_outcomes_are_deduplicated_by_episode(ingest):
    outcomes = make_outcomes("node-a", 1, 20, 0.5)
    ingest.report_outcomes(outcomes)
    ingest.report_outcomes(outcomes)

    assert ingest.stats()["outcomes"] == 20


def test_outcomes_are_acknowledged_so_the_node_can_drop_them(ingest):
    outcomes = make_outcomes("node-a", 1, 5, 0.5)
    stored = ingest.report_outcomes(outcomes)
    assert set(stored) == {o.episode_id for o in outcomes}


def test_outcomes_survive_a_shard_being_quarantined(ingest):
    """The measurement channel is not filtered by what happens to the data
    channel — a gate whose input depends on ingest luck is not a gate."""
    ingest.report_outcomes(make_outcomes("node-a", 1, 10, 0.7))
    bad = make_trajectory("e", node_id="node-a")
    bad.actions[0, 0] = 9.0
    manifest, payload = pack([bad])
    ingest.accept(manifest, payload)

    assert ingest.stats()["outcomes"] == 10
    assert ingest.stats()["quarantined"] == 1
