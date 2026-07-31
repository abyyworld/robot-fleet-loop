"""What the hub does before it believes anything a node sent it.

Data arriving from a fleet is the least trustworthy input in the system. It has
crossed a link, it was produced by software the hub may not have deployed yet,
on hardware with a clock nobody set, and it will be used to train the policy
that goes back down to every node. A single corrupted shard that reaches the
training set is a fleet-wide regression with no obvious cause.

**Rejected shards are quarantined, not dropped.** A drop is invisible; a
quarantine has a reason attached and can be looked at. The most valuable thing a
validation layer produces is not clean data — it is the list of what was dirty
and why, because that list is how you find the node with the failing sensor.

**Validation happens on the payload, not on the manifest.** A manifest is a
claim. The checks here are against the arrays themselves: shapes against the
declared contract, finiteness, action bounds, and a clock that is not absurd.

**Shards from a policy version the hub never published are refused.** Not
because they are necessarily bad, but because they cannot be attributed, and an
unattributable episode cannot be used to judge a release or to explain a
regression later.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from ..sim import RATE_CAP
from ..wire import SCHEMA_VERSION, EpisodeOutcome, ShardManifest, Trajectory, unpack_shard

# A node whose clock is a day out is a node whose data cannot be ordered against
# anything else. Generous, because clock drift on embedded hardware is normal.
MAX_CLOCK_SKEW = timedelta(hours=24)
MAX_AGE = timedelta(days=30)
ACTION_BOUND = RATE_CAP * 1.5


@dataclass
class Rejection:
    shard_id: str
    node_id: str
    reason: str
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class IngestResult:
    accepted: list[Trajectory] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    duplicate: bool = False

    @property
    def ok(self) -> bool:
        return not self.rejections and not self.duplicate


class Quarantine:
    """Everything the hub refused, with the reason, kept for inspection."""

    def __init__(self) -> None:
        self.items: list[Rejection] = []

    def add(self, rejection: Rejection) -> None:
        self.items.append(rejection)

    def by_reason(self) -> dict[str, int]:
        return dict(Counter(r.reason.split(":")[0] for r in self.items))

    def by_node(self) -> dict[str, int]:
        return dict(Counter(r.node_id for r in self.items))

    def __len__(self) -> int:
        return len(self.items)


class Validator:
    """Checks a shard against the contract the hub expects."""

    def __init__(
        self,
        obs_columns: list[str],
        action_columns: list[str],
        known_versions: set[int] | None = None,
    ) -> None:
        self.obs_columns = list(obs_columns)
        self.action_columns = list(action_columns)
        self.known_versions = known_versions if known_versions is not None else set()

    def check_manifest(self, manifest: ShardManifest) -> list[str]:
        """Cheap checks, before a payload is transferred at all."""
        problems: list[str] = []
        if manifest.schema_version > SCHEMA_VERSION:
            problems.append(
                f"schema:shard uses wire schema v{manifest.schema_version}, hub understands "
                f"v{SCHEMA_VERSION}"
            )
        if manifest.obs_columns != self.obs_columns:
            missing = set(self.obs_columns) ^ set(manifest.obs_columns)
            problems.append(f"contract:observation columns differ ({', '.join(sorted(missing))})")
        if manifest.action_columns != self.action_columns:
            problems.append("contract:action columns differ")
        if self.known_versions and manifest.policy_version not in self.known_versions:
            problems.append(
                f"provenance:policy v{manifest.policy_version} was never published by this hub, "
                "so these episodes cannot be attributed to a release"
            )
        problems.extend(self._check_clock(manifest))
        return problems

    def _check_clock(self, manifest: ShardManifest) -> list[str]:
        try:
            created = datetime.fromisoformat(manifest.created_at)
        except ValueError:
            return [f"clock:created_at is not a timestamp ({manifest.created_at!r})"]
        if created.tzinfo is None:
            return ["clock:created_at has no timezone"]

        now = datetime.now(timezone.utc)
        if created > now + MAX_CLOCK_SKEW:
            return [f"clock:created_at is {created - now} in the future"]
        if created < now - MAX_AGE:
            return [f"clock:shard is {now - created} old"]
        return []

    def check_payload(self, manifest: ShardManifest, trajectories: list[Trajectory]) -> list[str]:
        problems: list[str] = []
        if len(trajectories) != manifest.n_episodes:
            problems.append(
                f"shape:manifest claims {manifest.n_episodes} episodes, payload has "
                f"{len(trajectories)}"
            )

        for traj in trajectories:
            label = traj.episode_id[:8]
            if traj.steps == 0:
                problems.append(f"shape:episode {label} has no steps")
                continue
            if traj.observations.shape[1] != len(self.obs_columns):
                problems.append(
                    f"shape:episode {label} has {traj.observations.shape[1]} observation "
                    f"columns, contract says {len(self.obs_columns)}"
                )
            if traj.actions.shape[1] != len(self.action_columns):
                problems.append(
                    f"shape:episode {label} has {traj.actions.shape[1]} action columns, "
                    f"contract says {len(self.action_columns)}"
                )
            if not np.all(np.isfinite(traj.observations)):
                problems.append(f"values:episode {label} contains non-finite observations")
            if not np.all(np.isfinite(traj.actions)):
                problems.append(f"values:episode {label} contains non-finite actions")
            elif traj.actions.shape[1] >= 7:
                worst = float(np.abs(traj.actions[:, :7]).max())
                if worst > ACTION_BOUND:
                    # Beyond what the actuator can accept. Either the node is
                    # running something the hub did not send it, or the log is
                    # not what it claims to be. Either way it must not train.
                    problems.append(
                        f"values:episode {label} commands {worst:.3f} rad, above the "
                        f"{ACTION_BOUND:.3f} rad the actuator accepts"
                    )
        return problems


class Ingest:
    """Accepts shards and outcomes, deduplicating and quarantining."""

    def __init__(self, validator: Validator) -> None:
        self.validator = validator
        self.quarantine = Quarantine()
        self.shards: dict[str, ShardManifest] = {}
        self.trajectories: list[Trajectory] = []
        self.outcomes: dict[str, EpisodeOutcome] = {}
        self.bytes_received = 0

    # -- the cheap channel ---------------------------------------------------

    def report_outcomes(self, outcomes: list[EpisodeOutcome]) -> list[str]:
        """Store episode outcomes, returning the ids now held.

        Ids already present are returned too: the node is telling us about
        something it still holds, and confirming a duplicate is what
        at-least-once delivery needs.
        """
        stored: list[str] = []
        for outcome in outcomes:
            self.outcomes.setdefault(outcome.episode_id, outcome)
            stored.append(outcome.episode_id)
        return stored

    # -- the expensive channel ----------------------------------------------

    def have(self, shard_id: str) -> bool:
        return shard_id in self.shards

    def offer(self, manifests: list[ShardManifest]) -> tuple[list[str], list[str]]:
        """Answer "which of these do you want?" without a transfer.

        A manifest that fails its cheap checks is quarantined here, so the node
        does not spend a payload on a shard the hub was always going to refuse.
        """
        accept, already = [], []
        for manifest in manifests:
            if self.have(manifest.shard_id):
                already.append(manifest.shard_id)
                continue
            problems = self.validator.check_manifest(manifest)
            if problems:
                self.quarantine.add(
                    Rejection(manifest.shard_id, manifest.node_id, "; ".join(problems))
                )
                continue
            accept.append(manifest.shard_id)
        return accept, already

    def accept(self, manifest: ShardManifest, payload: bytes) -> IngestResult:
        """Validate and store a shard payload."""
        if self.have(manifest.shard_id):
            return IngestResult(duplicate=True)

        problems = self.validator.check_manifest(manifest)
        if problems:
            rejection = Rejection(manifest.shard_id, manifest.node_id, "; ".join(problems))
            self.quarantine.add(rejection)
            return IngestResult(rejections=[rejection])

        try:
            trajectories = unpack_shard(manifest, payload)
        except (ValueError, OSError) as exc:
            rejection = Rejection(manifest.shard_id, manifest.node_id, f"payload:{exc}")
            self.quarantine.add(rejection)
            return IngestResult(rejections=[rejection])

        problems = self.validator.check_payload(manifest, trajectories)
        if problems:
            rejection = Rejection(manifest.shard_id, manifest.node_id, "; ".join(problems))
            self.quarantine.add(rejection)
            return IngestResult(rejections=[rejection])

        self.shards[manifest.shard_id] = manifest
        self.trajectories.extend(trajectories)
        self.bytes_received += manifest.n_bytes
        return IngestResult(accepted=trajectories)

    # -- reporting -----------------------------------------------------------

    def stats(self) -> dict:
        return {
            "shards": len(self.shards),
            "episodes": len(self.trajectories),
            "outcomes": len(self.outcomes),
            "bytes_received": self.bytes_received,
            "quarantined": len(self.quarantine),
            "quarantine_by_reason": self.quarantine.by_reason(),
            "quarantine_by_node": self.quarantine.by_node(),
        }
