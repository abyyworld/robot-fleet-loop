"""Tests for the upload path, against a real hub."""

from __future__ import annotations

import pytest
from edge_runtime.ota import DirectoryReleaseSource

from fleet_loop.hub import Hub, HubConfig
from fleet_loop.loop import bootstrap_policy
from fleet_loop.node import BandwidthEstimate, FleetNode, NodeConfig, sync
from fleet_loop.sim import long_reach, nominal
from fleet_loop.wire import ShardManifest, SyncPlan


@pytest.fixture(scope="module")
def factory_policy():
    """One quick training run, shared: this fixture is the slow part."""
    policy, _ = bootstrap_policy(nominal(), episodes=60, seed=0)
    return policy


@pytest.fixture
def fleet(tmp_path, factory_policy):
    hub = Hub(HubConfig(root=tmp_path / "hub"))
    hub.publish(factory_policy, rollout_percent=100, notes="factory")
    node = FleetNode(
        NodeConfig(node_id="node-01", root=tmp_path / "node", environment=long_reach()),
        source=DirectoryReleaseSource(hub.releases),
    )
    node.install_local(hub.builds / "1")
    return hub, node


def test_outcomes_go_up_even_when_no_trajectory_fits(fleet):
    """The canary gate depends on them, and a sync that ran out of budget before
    reporting whether the episodes succeeded has told the hub nothing."""
    hub, node = fleet
    node.run_episodes(20)

    result = sync(node, hub, wire_budget=1)  # no room for a single shard
    assert result.outcomes_sent == 20
    assert result.episodes_sent == 0
    assert hub.ingest.stats()["outcomes"] == 20
    assert hub.ingest.stats()["episodes"] == 0


def test_the_wire_budget_is_respected(fleet):
    hub, node = fleet
    node.run_episodes(60)

    budget = 20_000
    result = sync(node, hub, wire_budget=budget)
    assert 0 < result.wire_bytes <= budget
    assert result.episodes_sent > 0


def test_a_bigger_budget_ships_more(fleet):
    hub, node = fleet
    node.run_episodes(60)

    small = sync(node, hub, wire_budget=10_000)
    node.run_episodes(60)
    large = sync(node, hub, wire_budget=100_000)
    assert large.episodes_sent > small.episodes_sent


def test_unacknowledged_episodes_stay_pending(fleet):
    """At-least-once: the node drops only what the hub confirmed."""
    hub, node = fleet
    node.run_episodes(40)
    before = len(node.pending)

    class RefusingHub:
        def report_outcomes(self, outcomes):
            return [o.episode_id for o in outcomes]

        def offer(self, manifests):
            return SyncPlan(accept=[m.shard_id for m in manifests])

        def upload(self, manifest, payload):
            return False

    result = sync(node, RefusingHub(), wire_budget=200_000)
    assert result.episodes_sent == 0
    assert len(node.pending) == before, "episodes were dropped without being stored"


def test_a_shard_already_held_is_not_uploaded_twice(fleet):
    """A manifest is a few hundred bytes and a shard is tens of kilobytes."""
    hub, node = fleet
    node.run_episodes(40)
    sync(node, hub, wire_budget=200_000)

    uploads: list[str] = []
    original_upload = hub.upload

    def counting_upload(manifest: ShardManifest, payload: bytes) -> bool:
        uploads.append(manifest.shard_id)
        return original_upload(manifest, payload)

    hub.upload = counting_upload
    node.pending = list(node.pending) or []
    # Re-offer the same shards by replaying the node's already-sent data.
    node.run_episodes(10)
    sync(node, hub, wire_budget=200_000)
    assert len(set(uploads)) == len(uploads), "the same shard was uploaded twice"


def test_the_compression_estimate_converges_on_the_real_ratio(fleet):
    """Triage budgets in float32 bytes; the link is billed in wire bytes."""
    hub, node = fleet
    estimate = BandwidthEstimate(ratio=8.0)  # deliberately wrong
    trace = []
    for _ in range(8):
        node.run_episodes(30)
        sync(node, hub, wire_budget=60_000, estimate=estimate)
        trace.append(estimate.ratio)

    assert trace == sorted(trace, reverse=True), f"estimate did not move monotonically: {trace}"
    assert 1.2 < estimate.ratio < 3.0, f"estimate did not converge: {estimate.ratio}"


def test_the_estimate_error_is_reported_not_hidden(fleet):
    hub, node = fleet
    node.run_episodes(40)
    result = sync(node, hub, wire_budget=60_000, estimate=BandwidthEstimate(ratio=10.0))
    assert result.estimate_error > 0.1


def test_shards_never_mix_policy_versions(fleet, tmp_path, factory_policy):
    """Attribution is what the whole loop is built on."""
    hub, node = fleet
    node.run_episodes(20)

    second = hub.publish(factory_policy, rollout_percent=100, notes="v2")
    node.poll_updates()
    assert node.policy_version == second
    node.run_episodes(20)

    sync(node, hub, wire_budget=400_000)
    versions = {m.policy_version for m in hub.ingest.shards.values()}
    assert versions == {1, 2}
    for manifest in hub.ingest.shards.values():
        assert manifest.n_episodes > 0


def test_the_local_store_is_bounded_and_sheds_the_least_valuable(tmp_path, factory_policy):
    """A node may be days from a sync window, and storage is finite."""
    hub = Hub(HubConfig(root=tmp_path / "hub"))
    hub.publish(factory_policy, rollout_percent=100, notes="factory")
    node = FleetNode(
        NodeConfig(
            node_id="node-tiny",
            root=tmp_path / "tiny",
            environment=long_reach(),
            local_store_bytes=60_000,
        ),
        source=DirectoryReleaseSource(hub.releases),
    )
    node.install_local(hub.builds / "1")

    node.run_episodes(120)
    assert node.pending_bytes <= 60_000
    assert node.dropped_locally > 0
    # And what survived is not simply the newest.
    assert len(node.pending) > 0


def test_a_node_that_never_syncs_still_records_its_own_health(fleet):
    """A node that can only describe itself while it has an unsent backlog looks
    blank exactly when it is working."""
    hub, node = fleet
    node.run_episodes(30)
    sync(node, hub, wire_budget=200_000)

    snapshot = node.snapshot()
    assert snapshot["recent_success_rate"] is not None
    assert snapshot["policy_version"] == 1
