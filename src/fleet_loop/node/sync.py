"""Selective, bandwidth-aware upload.

The budget is in **wire bytes**, because that is what a data plan is priced in
and what a sync window is limited by. Triage, however, ranks episodes by their
in-memory float32 size, which is what it can see. The gap between the two is
about 2x — see :mod:`fleet_loop.wire` for where that comes from and why it is not
the 3x that generic compression is usually assumed to give — so the node keeps a
running estimate of its own ratio, budgets triage against it, and then enforces
the true wire budget after packing. The estimate's error is reported rather than
hidden: a node whose ratio has moved is a node whose data has changed character,
and that is worth knowing before it silently overruns its budget.

The order of operations matters:

1. **Outcomes go first, unconditionally.** They are ~80 bytes each and the
   canary gate depends on them. A sync that ran out of budget before reporting
   whether the episodes succeeded has told the hub nothing it can act on.
2. **Manifests are offered before payloads.** The hub replies with the subset it
   does not already have. A manifest is a few hundred bytes and a shard is tens
   of kilobytes, so a node retrying after a lost response pays almost nothing
   for the retry.
3. **Payloads are uploaded, and pending episodes are dropped only on
   acknowledgement.** Same contract as the telemetry up-link: at-least-once,
   deduplicated by content hash at the far end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..wire import EpisodeOutcome, ShardManifest, SyncPlan, Trajectory, pack_shard
from .runner import FleetNode
from .triage import TriageScore, select_within_budget

DEFAULT_EPISODES_PER_SHARD = 8


class HubEndpoint(Protocol):
    """What a node needs from the other end.

    In-process here, because the interesting part of this repo is the loop and
    not a second HTTP server — `edge-policy-runtime` already demonstrates both links
    over real HTTP, and this interface is narrow enough to put behind one.
    """

    def report_outcomes(self, outcomes: list[EpisodeOutcome]) -> list[str]:
        """Accept episode outcomes; return the ids durably stored."""
        ...

    def offer(self, manifests: list[ShardManifest]) -> SyncPlan:
        """Given shard descriptions, say which payloads are wanted."""
        ...

    def upload(self, manifest: ShardManifest, payload: bytes) -> bool:
        """Accept a shard payload. False means it was not stored."""
        ...


@dataclass
class BandwidthEstimate:
    """A node's running estimate of how well its own data compresses."""

    ratio: float = 2.0
    alpha: float = 0.3

    def triage_budget(self, wire_budget: int) -> int:
        return int(wire_budget * self.ratio)

    def observe(self, raw_bytes: int, wire_bytes: int) -> float:
        if wire_bytes <= 0:
            return self.ratio
        observed = raw_bytes / wire_bytes
        self.ratio = (1 - self.alpha) * self.ratio + self.alpha * observed
        return observed


@dataclass
class SyncResult:
    node_id: str
    policy_version: int
    outcomes_sent: int
    episodes_offered: int
    episodes_sent: int
    wire_bytes: int
    wire_budget: int
    duplicate_shards: int
    triage: dict = field(default_factory=dict)
    estimate_error: float = 0.0

    @property
    def budget_used(self) -> float:
        return self.wire_bytes / max(self.wire_budget, 1)

    def summary(self) -> str:
        return (
            f"{self.node_id}: {self.episodes_sent}/{self.triage.get('considered', 0)} episodes, "
            f"{self.wire_bytes / 1024:.0f} KiB of {self.wire_budget / 1024:.0f} KiB budget "
            f"({self.budget_used:.0%})"
        )


def sync(
    node: FleetNode,
    hub: HubEndpoint,
    wire_budget: int,
    *,
    estimate: BandwidthEstimate | None = None,
    episodes_per_shard: int = DEFAULT_EPISODES_PER_SHARD,
    rng: np.random.Generator | None = None,
) -> SyncResult:
    """Report outcomes, then ship as much of the valuable data as fits."""
    estimate = estimate or BandwidthEstimate()

    # 1. The cheap channel, always.
    outcomes = node.take_outcomes()
    stored = set(hub.report_outcomes(outcomes)) if outcomes else set()
    # Anything the hub did not confirm goes back on the queue.
    node.outcomes = [o for o in outcomes if o.episode_id not in stored] + node.outcomes

    if not node.pending:
        return SyncResult(
            node_id=node.config.node_id,
            policy_version=node.policy_version,
            outcomes_sent=len(stored),
            episodes_offered=0,
            episodes_sent=0,
            wire_bytes=0,
            wire_budget=wire_budget,
            duplicate_shards=0,
        )

    # 2. Triage against the raw-byte budget implied by the wire budget.
    selected, triage_stats = select_within_budget(
        node.pending,
        node.coverage,
        estimate.triage_budget(wire_budget),
        rng=rng,
    )
    if not selected:
        return SyncResult(
            node_id=node.config.node_id,
            policy_version=node.policy_version,
            outcomes_sent=len(stored),
            episodes_offered=0,
            episodes_sent=0,
            wire_bytes=0,
            wire_budget=wire_budget,
            duplicate_shards=0,
            triage=triage_stats,
        )

    # 3. Pack into shards, most valuable first, so that truncation at the wire
    #    budget drops the least valuable data rather than an arbitrary tail.
    shards = _pack(node, selected, episodes_per_shard)

    raw_bytes = sum(s.nbytes for s in selected)
    packed_bytes = sum(s.manifest.n_bytes for s in shards)
    observed_ratio = estimate.observe(raw_bytes, packed_bytes)

    kept: list[Shard] = []
    spent = 0
    for shard in shards:
        if spent + shard.manifest.n_bytes > wire_budget:
            continue
        kept.append(shard)
        spent += shard.manifest.n_bytes

    # 4. Offer manifests; upload only what the hub does not already have.
    plan = hub.offer([s.manifest for s in kept])
    wanted = set(plan.accept)
    duplicates = len(plan.already_have)

    sent_bytes = 0
    sent_episodes = 0
    acknowledged: set[str] = set(plan.already_have)
    for shard in kept:
        if shard.manifest.shard_id not in wanted:
            continue
        if hub.upload(shard.manifest, shard.payload):
            acknowledged.add(shard.manifest.shard_id)
            sent_bytes += shard.manifest.n_bytes
            sent_episodes += shard.manifest.n_episodes

    # 5. Drop only what the hub confirmed. Everything else stays pending and is
    #    reconsidered next window, where it competes on value like anything else.
    shipped = {
        episode_id
        for shard in kept
        if shard.manifest.shard_id in acknowledged
        for episode_id in shard.episode_ids
    }
    node.pending = [t for t in node.pending if t.episode_id not in shipped]

    return SyncResult(
        node_id=node.config.node_id,
        policy_version=node.policy_version,
        outcomes_sent=len(stored),
        episodes_offered=sum(s.manifest.n_episodes for s in kept),
        episodes_sent=sent_episodes,
        wire_bytes=sent_bytes,
        wire_budget=wire_budget,
        duplicate_shards=duplicates,
        triage=triage_stats,
        estimate_error=abs(observed_ratio - estimate.ratio) / max(observed_ratio, 1e-9),
    )


@dataclass
class Shard:
    manifest: ShardManifest
    payload: bytes
    episode_ids: list[str]


def _pack(node: FleetNode, selected: list[TriageScore], episodes_per_shard: int) -> list[Shard]:
    """Group selected episodes into shards, most valuable shard first.

    Shards are grouped by policy version because a shard that mixes versions
    cannot be attributed to the release that produced it — and attribution is
    what the whole loop is built on.
    """
    by_version: dict[int, list[TriageScore]] = {}
    for item in sorted(selected, key=lambda s: s.value_per_byte, reverse=True):
        by_version.setdefault(item.trajectory.policy_version, []).append(item)

    shards: list[Shard] = []
    for group in by_version.values():
        for start in range(0, len(group), episodes_per_shard):
            chunk = group[start : start + episodes_per_shard]
            reasons: dict[str, int] = {}
            for item in chunk:
                reasons[item.top_reason()] = reasons.get(item.top_reason(), 0) + 1
            trajectories = [item.trajectory for item in chunk]
            manifest, payload = pack_shard(
                trajectories,
                obs_columns=node.env.obs_columns,
                action_columns=list(node.runtime.policy.action_columns),
                value=sum(item.value for item in chunk),
                reasons=reasons,
            )
            shards.append(Shard(manifest, payload, [t.episode_id for t in trajectories]))

    shards.sort(key=lambda s: s.manifest.value, reverse=True)
    return shards


def trajectories_of(shards: list[tuple[ShardManifest, bytes]]) -> list[Trajectory]:
    from ..wire import unpack_shard

    return [t for m, p in shards for t in unpack_shard(m, p)]
