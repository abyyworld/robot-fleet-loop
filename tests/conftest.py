"""Shared fixtures: cheap synthetic trajectories, and a real small fleet."""

from __future__ import annotations

import numpy as np
import pytest
from edge_runtime.policy import ACTION_COLUMNS, default_obs_columns

from fleet_loop.wire import EpisodeOutcome, Trajectory

OBS_COLUMNS = default_obs_columns()
ACTIONS = list(ACTION_COLUMNS)


def make_trajectory(
    episode_id: str = "e0",
    *,
    node_id: str = "node-a",
    policy_version: int = 1,
    success: bool = True,
    steps: int = 40,
    flags: list[str] | None = None,
    goal: float = 0.0,
    seed: int = 0,
    finished_at: str = "2026-01-01T00:00:00+00:00",
) -> Trajectory:
    rng = np.random.default_rng(seed)
    # Smooth, as a real trajectory is: joints integrate small deltas. Pure noise
    # would make the compression and novelty tests measure nothing real.
    actions = np.clip(
        np.cumsum(rng.normal(0, 0.01, (steps, len(ACTIONS))), axis=0), -0.15, 0.15
    ).astype(np.float32)
    joints = np.cumsum(actions[:, :7], axis=0) + goal
    observations = np.zeros((steps, len(OBS_COLUMNS)), dtype=np.float32)
    observations[:, :7] = joints
    observations[:, 7:14] = goal - joints
    observations[:, 14] = actions[:, 7]
    observations[:, 15] = np.linspace(0, 1, steps)
    observations[:, 16] = np.linalg.norm(goal - joints, axis=1)
    observations[:, 17] = np.linalg.norm(actions[:, :7], axis=1)
    return Trajectory(
        episode_id=episode_id,
        node_id=node_id,
        policy_version=policy_version,
        observations=observations,
        actions=actions,
        success=success,
        final_distance=0.02 if success else 0.9,
        flags=flags or [],
        finished_at=finished_at,
    )


def make_outcomes(
    node_id: str, policy_version: int, n: int, success_rate: float
) -> list[EpisodeOutcome]:
    successes = int(round(n * success_rate))
    return [
        EpisodeOutcome(
            episode_id=f"{node_id}-{policy_version}-{i}",
            node_id=node_id,
            policy_version=policy_version,
            finished_at="2026-01-01T00:00:00+00:00",
            success=i < successes,
            steps=40,
            final_distance=0.02 if i < successes else 0.9,
        )
        for i in range(n)
    ]


@pytest.fixture
def trajectories():
    return [
        make_trajectory(f"e{i}", success=i % 3 != 0, seed=i, goal=(i % 5) * 0.4) for i in range(24)
    ]
