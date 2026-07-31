"""What crosses the link, in both directions.

Two channels go up, and keeping them separate is the single most useful decision
in this repo.

**Outcomes are cheap and always sent.** One episode summary is about eighty
bytes: which node, which policy version, did it succeed, how many steps. Every
episode produces one, and no triage decides whether it goes. The canary gate
depends on them, and a gate whose input is filtered by a value heuristic is
measuring the heuristic.

**Trajectories are expensive and mostly not sent.** One episode of
observations and actions is tens of kilobytes. Which of them go up is the whole
subject of :mod:`fleet_loop.node.triage`, and it is a decision about two scarce
resources at once: bandwidth on the wire, and a human's time at the other end.

Conflating the two — deciding what to measure with the same rule that decides
what to collect — is how a fleet ends up unable to tell whether a release helped,
because the only episodes it has records of are the ones a heuristic found
interesting.

Shards are content-addressed by the SHA-256 of their payload, so a shard
uploaded twice is detected rather than duplicated, and two nodes that hit the
same failure do not each pay to ship it.

**Payloads are float16 in two concatenated arrays, and generic compression does
almost nothing.** That was measured rather than assumed, and the measurement
changed the design twice.

First, deflate on float32 trajectory data comes out at roughly 1.0x — sometimes
below 1 once the archive framing is counted. Real-valued sensor data has
high-entropy low-order mantissa bits and there is nothing for a dictionary coder
to find. The "trajectories compress about 3x" assumption this was built on was
simply wrong.

Second, storing one array pair *per episode* meant a zip member header per
episode, and at eight episodes of ten steps the framing was a fifth of the
shard. Concatenating into one observation array and one action array plus a
length index fixed that.

What ends up on the wire is 1.6–2.2x smaller than the in-memory float32,
depending on episode length, and the saving is almost entirely the dtype. The
precision cost is bounded and tested: worst-case round-trip error is 1.6e-3 rad
on a joint angle whose success tolerance is 0.10 rad, and 6e-5 rad on an action
whose rate cap is 0.15. A robot's encoders do not resolve better than that, and
shipping bits below the sensor's noise floor is paying to transmit noise.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- the cheap channel -------------------------------------------------------


class EpisodeOutcome(BaseModel):
    """What every episode reports, whether or not its trajectory is uploaded."""

    episode_id: str
    node_id: str
    policy_version: int
    finished_at: str
    success: bool
    steps: int
    final_distance: float
    flags: list[str] = Field(default_factory=list)

    def nbytes(self) -> int:
        return len(self.model_dump_json().encode())


# -- the expensive channel ---------------------------------------------------


@dataclass
class Trajectory:
    """One episode's observations and actions, as the node recorded them."""

    episode_id: str
    node_id: str
    policy_version: int
    observations: np.ndarray  # (T, obs_dim), float32
    actions: np.ndarray  # (T, action_dim), float32
    success: bool
    final_distance: float
    flags: list[str] = field(default_factory=list)
    finished_at: str = field(default_factory=now)

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        if len(self.observations) != len(self.actions):
            raise ValueError(
                f"episode {self.episode_id}: {len(self.observations)} observations "
                f"but {len(self.actions)} actions"
            )

    @property
    def steps(self) -> int:
        return len(self.observations)

    @property
    def nbytes(self) -> int:
        return self.observations.nbytes + self.actions.nbytes

    def outcome(self) -> EpisodeOutcome:
        return EpisodeOutcome(
            episode_id=self.episode_id,
            node_id=self.node_id,
            policy_version=self.policy_version,
            finished_at=self.finished_at,
            success=self.success,
            steps=self.steps,
            final_distance=self.final_distance,
            flags=list(self.flags),
        )


class ShardManifest(BaseModel):
    """The description of an upload, separate from its payload.

    The hub reads this before it reads a single byte of data, which is what makes
    "do I already have this?" and "is this from a policy version I know about?"
    answerable without a transfer.
    """

    schema_version: int = SCHEMA_VERSION
    shard_id: str  # sha256 of the payload bytes
    node_id: str
    policy_version: int
    created_at: str
    n_episodes: int
    n_steps: int
    n_bytes: int
    obs_columns: list[str]
    action_columns: list[str]

    # Why triage selected this, carried up so hub-side analysis of what the
    # heuristic is actually shipping does not require re-deriving it.
    value: float = 0.0
    reasons: dict[str, int] = Field(default_factory=dict)
    success_rate: float = 0.0

    @property
    def bytes_per_episode(self) -> float:
        return self.n_bytes / max(self.n_episodes, 1)


def pack_shard(
    trajectories: list[Trajectory],
    *,
    obs_columns: list[str],
    action_columns: list[str],
    value: float = 0.0,
    reasons: dict[str, int] | None = None,
) -> tuple[ShardManifest, bytes]:
    """Serialise trajectories into one blob and describe it.

    Stored as float16. See the module docstring for the measurement that led
    there, and for the precision bound it costs.
    """
    if not trajectories:
        raise ValueError("a shard with no episodes is not worth an upload")

    node_ids = {t.node_id for t in trajectories}
    versions = {t.policy_version for t in trajectories}
    if len(node_ids) != 1 or len(versions) != 1:
        # A shard mixing nodes or policy versions cannot be attributed, and
        # attribution is what the whole loop is built on.
        raise ValueError("a shard must come from one node running one policy version")

    # One concatenated array per field plus a length index, rather than two
    # arrays per episode. A zip member costs a header, and at eight episodes of
    # ten steps the headers were a fifth of the shard — the framing was larger
    # than anything the encoding could save.
    index = [
        {
            "episode_id": traj.episode_id,
            "success": bool(traj.success),
            "final_distance": float(traj.final_distance),
            "flags": list(traj.flags),
            "finished_at": traj.finished_at,
        }
        for traj in trajectories
    ]
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        observations=np.concatenate([t.observations for t in trajectories]).astype(np.float16),
        actions=np.concatenate([t.actions for t in trajectories]).astype(np.float16),
        lengths=np.array([t.steps for t in trajectories], dtype=np.int32),
        __index__=np.frombuffer(json.dumps(index).encode(), dtype=np.uint8),
    )
    payload = buffer.getvalue()

    manifest = ShardManifest(
        shard_id=hashlib.sha256(payload).hexdigest(),
        node_id=next(iter(node_ids)),
        policy_version=next(iter(versions)),
        created_at=now(),
        n_episodes=len(trajectories),
        n_steps=sum(t.steps for t in trajectories),
        n_bytes=len(payload),
        obs_columns=list(obs_columns),
        action_columns=list(action_columns),
        value=value,
        reasons=reasons or {},
        success_rate=sum(t.success for t in trajectories) / len(trajectories),
    )
    return manifest, payload


def unpack_shard(manifest: ShardManifest, payload: bytes) -> list[Trajectory]:
    """Reconstruct trajectories, checking the payload is the one described."""
    actual = hashlib.sha256(payload).hexdigest()
    if actual != manifest.shard_id:
        raise ValueError(
            f"shard payload hashes to {actual[:12]} but the manifest says "
            f"{manifest.shard_id[:12]} — truncated transfer, or the wrong payload"
        )

    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        index = json.loads(bytes(data["__index__"]).decode())
        bounds = np.concatenate([[0], np.cumsum(data["lengths"])])
        observations, actions = data["observations"], data["actions"]
        return [
            Trajectory(
                episode_id=entry["episode_id"],
                node_id=manifest.node_id,
                policy_version=manifest.policy_version,
                observations=observations[bounds[i] : bounds[i + 1]],
                actions=actions[bounds[i] : bounds[i + 1]],
                success=entry["success"],
                final_distance=entry["final_distance"],
                flags=entry["flags"],
                finished_at=entry["finished_at"],
            )
            for i, entry in enumerate(index)
        ]


# -- what comes back down ----------------------------------------------------


class SyncPlan(BaseModel):
    """The hub's answer to "what should I send you?", before anything is sent.

    The node proposes shards by id and size; the hub replies with the subset it
    does not already have. Two nodes that hit the same failure mode produce
    byte-identical shards only rarely, but a node that retries an upload after a
    lost response produces exactly that — and a round trip of manifests is far
    cheaper than a round trip of payloads.
    """

    accept: list[str] = Field(default_factory=list)
    already_have: list[str] = Field(default_factory=list)
    byte_budget_remaining: int = 0
