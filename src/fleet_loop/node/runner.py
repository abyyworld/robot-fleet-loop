"""A fleet node: runs the policy it was given, and decides what is worth telling.

The node is `edge-policy-runtime`'s device with two things added — episodes, and a
local store of what happened during them. The down-link, the health gate and the
edge-case detectors are that repo's; nothing here reimplements them.

**The local store is bounded and evicts by value, not by age.** Storage on a
node is finite and a node may be days from a sync window. A ring buffer would
discard the failure that happened on Monday to make room for Thursday's routine
successes, which inverts the point of collecting at all. The same score that
decides what to upload decides what to keep.

**The node never uploads on its own schedule.** Sync is called with a byte
budget by whatever knows about the link — a scheduler that has seen the data
plan, an operator, a docking station. A node that decides for itself when to
spend bandwidth is a node that will do it at the worst possible moment.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from edge_runtime import bundle as bundle_mod
from edge_runtime.ota import OTAClient, ReleaseSource
from edge_runtime.policy import Policy
from edge_runtime.runtime import InferenceRuntime
from edge_runtime.telemetry import EdgeCaseDetector

from ..sim import NodeEnvironment, ReachEnv, nominal
from ..wire import EpisodeOutcome, Trajectory
from .triage import Coverage, score


@dataclass
class NodeConfig:
    node_id: str
    root: Path
    environment: NodeEnvironment = field(default_factory=nominal)
    channel: str = "stable"
    budget_ms: float = 20.0
    # Roughly two hundred episodes of trajectory, which is a few megabytes.
    local_store_bytes: int = 4_000_000


class FleetNode:
    """One node in the fleet."""

    def __init__(
        self,
        config: NodeConfig,
        source: ReleaseSource | None = None,
    ) -> None:
        self.config = config
        self.env = ReachEnv(env=config.environment)
        self.ota = OTAClient(
            Path(config.root) / "ota",
            config.node_id,
            source,
            channel=config.channel,
            device_obs_columns=self.env.obs_columns,
        )
        self.coverage = Coverage()
        self.pending: list[Trajectory] = []
        # `outcomes` is the outbound queue and is drained by sync. `recent` is
        # what the node knows about itself, and it must not disappear the moment
        # the hub acknowledges it — a node that can only describe its own health
        # while it has an unsent backlog is a node that looks blank when it is
        # working perfectly.
        self.outcomes: list[EpisodeOutcome] = []
        self.recent: deque[EpisodeOutcome] = deque(maxlen=200)
        self.runtime: InferenceRuntime | None = None
        self.detector = EdgeCaseDetector()
        self.dropped_locally = 0
        self._load_current()

    # -- policy lifecycle ----------------------------------------------------

    def _load_current(self) -> None:
        policy = self.ota.current_policy()
        if policy is None:
            self.runtime = None
            return
        self.runtime = InferenceRuntime(policy, budget_ms=self.config.budget_ms)
        self.runtime.warmup()
        self._calibrate_detector(policy)

    def _calibrate_detector(self, policy: Policy) -> None:
        """Refit the out-of-distribution detector to this policy on this node.

        Both halves matter. A detector calibrated on the previous policy fires
        constantly for the first hour after an update; one calibrated on the
        fleet average never fires on the node whose conditions are unusual —
        which is the node worth hearing from.
        """
        scratch = ReachEnv(env=self.config.environment)
        observations = []
        obs = scratch.reset(seed=self.config.environment.seed + 7717)
        for _ in range(512):
            observations.append(obs)
            obs, done, _ = scratch.step(policy.forward(obs))
            if done:
                obs = scratch.reset()
        self.detector = EdgeCaseDetector.calibrated_on(np.asarray(observations))

    def install_local(self, bundle_dir: Path | str) -> int:
        """Boot the node onto a factory bundle."""
        bundle_dir = Path(bundle_dir)
        manifest = bundle_mod.verify(bundle_dir)
        dest = self.ota.bundle_dir(manifest.version)
        if not dest.exists():
            dest.mkdir(parents=True)
            for item in bundle_dir.iterdir():
                if item.is_file():
                    dest.joinpath(item.name).write_bytes(item.read_bytes())
        result = self.ota.activate(manifest.version)
        if not result.passed:
            raise RuntimeError(f"factory bundle failed this node's health gate: {result.summary()}")
        self._load_current()
        return manifest.version

    def poll_updates(self) -> dict:
        """One OTA cycle. Reloads the policy and recalibrates if it changed."""
        outcome = self.ota.poll()
        if outcome.get("updated"):
            self._load_current()
        return outcome

    @property
    def policy_version(self) -> int:
        return self.ota.state.current_version

    # -- running -------------------------------------------------------------

    def run_episodes(self, n: int) -> dict:
        """Run `n` episodes, recording a trajectory and an outcome for each."""
        if self.runtime is None:
            raise RuntimeError(f"node {self.config.node_id} has no active policy")

        successes = 0
        for _ in range(n):
            trajectory = self._run_one()
            outcome = trajectory.outcome()
            self.outcomes.append(outcome)
            self.recent.append(outcome)
            self.pending.append(trajectory)
            successes += trajectory.success
        self._enforce_store_budget()

        stats = self.runtime.stats()
        return {
            "node_id": self.config.node_id,
            "policy_version": self.policy_version,
            "episodes": n,
            "success_rate": successes / n if n else 0.0,
            "pending_episodes": len(self.pending),
            "pending_bytes": self.pending_bytes,
            "dropped_locally": self.dropped_locally,
            "latency_p99_ms": stats.p99_ms,
            "deadline_misses": stats.deadline_misses,
        }

    def _run_one(self) -> Trajectory:
        assert self.runtime is not None
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        flags: list[str] = []

        obs = self.env.reset()
        success = False
        while True:
            record = self.runtime.step(obs)
            for reason in self.detector.inspect(obs, record):
                flags.append(reason)

            observations.append(obs)
            obs, done, success = self.env.step(record.action)
            # What the actuator accepted, not what the policy emitted. See
            # ReachEnv.step: the difference is what makes these logs trainable.
            actions.append(self.env.commanded)
            if done:
                break

        return Trajectory(
            episode_id=uuid.uuid4().hex[:16],
            node_id=self.config.node_id,
            policy_version=self.policy_version,
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            success=success,
            final_distance=self.env.distance,
            flags=flags,
        )

    # -- local storage -------------------------------------------------------

    @property
    def pending_bytes(self) -> int:
        return sum(t.nbytes for t in self.pending)

    def _enforce_store_budget(self) -> None:
        """Drop the least valuable pending episodes when storage runs out.

        Evaluated against a scratch copy of the coverage table so that scoring
        for eviction does not tell the node it has already shipped things it has
        not.
        """
        if self.pending_bytes <= self.config.local_store_bytes:
            return

        scratch = Coverage(bins=self.coverage.bins, counts=self.coverage.counts.copy())
        ranked = sorted(self.pending, key=lambda t: score(t, scratch).value_per_byte, reverse=True)
        kept: list[Trajectory] = []
        total = 0
        for trajectory in ranked:
            if total + trajectory.nbytes > self.config.local_store_bytes:
                self.dropped_locally += 1
                continue
            kept.append(trajectory)
            total += trajectory.nbytes
        # Back into chronological order; the store is value-bounded, not
        # value-ordered, and downstream code should not depend on the ranking.
        order = {t.episode_id: i for i, t in enumerate(self.pending)}
        self.pending = sorted(kept, key=lambda t: order[t.episode_id])

    def take_outcomes(self) -> list[EpisodeOutcome]:
        outcomes, self.outcomes = self.outcomes, []
        return outcomes

    # -- reporting -----------------------------------------------------------

    def snapshot(self) -> dict:
        stats = self.runtime.stats() if self.runtime else None
        recent = list(self.recent)[-100:]
        return {
            "node_id": self.config.node_id,
            "environment": self.config.environment.name,
            "policy_version": self.policy_version,
            "quarantined": list(self.ota.state.quarantined),
            "updates_frozen": self.ota.state.updates_frozen,
            "pending_episodes": len(self.pending),
            "pending_bytes": self.pending_bytes,
            "dropped_locally": self.dropped_locally,
            "distinct_regions_sent": self.coverage.distinct_regions(),
            "latency_p99_ms": round(stats.p99_ms, 3) if stats else None,
            "deadline_misses": stats.deadline_misses if stats else 0,
            "recent_success_rate": (
                sum(o.success for o in recent) / len(recent) if recent else None
            ),
        }
