"""Turning accepted shards into a training set that can be pointed at later.

**A dataset is content-addressed by the shards in it.** `dataset_hash` is the
SHA-256 of the sorted shard ids plus the labelling rule. Two assemblies of the
same shards produce the same hash, and every checkpoint trained from it can name
the data that produced it — the same join key the training pipeline and the
evaluation harness upstream use.

**Failures are relabelled; successes are not.** This is the part worth being
precise about, because it is where a fleet loop is usually vague.

- A **failed** episode's actions are, by definition, actions that did not work.
  Training on them teaches the policy to reproduce the failure. What is valuable
  is the *states* it visited: those are the states the current policy cannot
  handle, and what they need is a correct action attached to each. That is the
  DAgger step, and in a real fleet the corrections come from a teleoperator.
  :func:`fleet_loop.sim.relabel` stands in for that person.
- A **successful** episode's actions did work, in that node's conditions,
  including whatever miscalibration that node has. They are demonstrations and
  are used as recorded.

This is also why the node uploads actions and not only observations: without the
node's own actions there is nothing to keep from the successes, and no way to
measure how far the policy was from the correction where it failed.
:attr:`DatasetVersion.mean_correction` is that measurement, and it is the most
useful single number the hub produces — it says how wrong the deployed policy is
in the states it is actually failing in, in radians.

**The set is windowed.** Training on every episode ever collected means the
first policy's failures outnumber the current one's forever, and the model spends
its capacity on states no deployed policy visits any more. The window is over
episodes, most recent first, and the policy-version mix is recorded so the drift
is visible rather than assumed away.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from ..sim import relabel
from ..wire import Trajectory

DEFAULT_WINDOW = 4000  # episodes


class DatasetVersion(BaseModel):
    """The record of what a training set was made of."""

    dataset_hash: str
    created_at: str
    n_shards: int
    n_episodes: int
    n_steps: int
    shard_ids: list[str] = Field(default_factory=list)

    episodes_by_node: dict[str, int] = Field(default_factory=dict)
    episodes_by_policy_version: dict[str, int] = Field(default_factory=dict)
    success_rate: float = 0.0
    relabelled_episodes: int = 0
    demonstrated_episodes: int = 0
    mean_correction: float = 0.0
    parent: str | None = None

    def summary(self) -> str:
        nodes = ", ".join(f"{k}={v}" for k, v in sorted(self.episodes_by_node.items()))
        return (
            f"{self.dataset_hash[:12]}  {self.n_episodes} episodes / {self.n_steps} steps  "
            f"[{nodes}]  {self.relabelled_episodes} relabelled, "
            f"mean correction {self.mean_correction:.3f} rad"
        )

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2), encoding="utf-8")
        return path


@dataclass
class TrainingSet:
    version: DatasetVersion
    observations: np.ndarray  # (N, obs_dim)
    actions: np.ndarray  # (N, action_dim) — the targets, after relabelling

    def __len__(self) -> int:
        return len(self.observations)


def assemble(
    trajectories: list[Trajectory],
    shard_ids: list[str],
    *,
    window: int = DEFAULT_WINDOW,
    parent: str | None = None,
    relabel_failures: bool = True,
) -> TrainingSet:
    """Build a training set from accepted episodes, and describe it."""
    if not trajectories:
        raise ValueError("cannot assemble a dataset from no episodes")

    # Most recent first, then truncate. `finished_at` rather than arrival order,
    # because shards arrive when a node happens to get a sync window and that
    # says nothing about when the episode happened.
    ordered = sorted(trajectories, key=lambda t: t.finished_at, reverse=True)[:window]

    observation_blocks: list[np.ndarray] = []
    action_blocks: list[np.ndarray] = []
    corrections: list[float] = []
    relabelled = demonstrated = 0

    for traj in ordered:
        observation_blocks.append(traj.observations)
        if traj.success or not relabel_failures:
            action_blocks.append(traj.actions)
            demonstrated += 1
        else:
            expert = relabel(traj.observations)
            action_blocks.append(expert)
            corrections.append(float(np.abs(expert[:, :7] - traj.actions[:, :7]).mean()))
            relabelled += 1

    observations = np.concatenate(observation_blocks).astype(np.float32)
    actions = np.concatenate(action_blocks).astype(np.float32)

    digest = hashlib.sha256()
    for shard_id in sorted(shard_ids):
        digest.update(shard_id.encode())
    digest.update(f"|window={window}|relabel={relabel_failures}".encode())

    version = DatasetVersion(
        dataset_hash=digest.hexdigest(),
        created_at=datetime.now(timezone.utc).isoformat(),
        n_shards=len(shard_ids),
        n_episodes=len(ordered),
        n_steps=len(observations),
        shard_ids=sorted(shard_ids),
        episodes_by_node=dict(Counter(t.node_id for t in ordered)),
        episodes_by_policy_version=dict(Counter(str(t.policy_version) for t in ordered)),
        success_rate=sum(t.success for t in ordered) / len(ordered),
        relabelled_episodes=relabelled,
        demonstrated_episodes=demonstrated,
        mean_correction=float(np.mean(corrections)) if corrections else 0.0,
        parent=parent,
    )
    return TrainingSet(version, observations, actions)


def coverage_gain(new: DatasetVersion, previous: DatasetVersion | None) -> float:
    """Fraction of the new dataset's shards that the previous one did not have.

    The retrain trigger uses this. New episodes that are all from shards already
    trained on are not a reason to spend a training run and an evaluation cycle.
    """
    if previous is None:
        return 1.0
    fresh = set(new.shard_ids) - set(previous.shard_ids)
    return len(fresh) / max(len(new.shard_ids), 1)
