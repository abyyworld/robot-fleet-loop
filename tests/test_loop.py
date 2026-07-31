"""The claim, asserted end to end.

These are the slow tests. They exist because every other test in this suite
checks a part, and the thing worth proving is that the parts close a loop: a
fleet gets better from its own data, and a release that is worse comes back.
"""

from __future__ import annotations

import json

import pytest
from edge_runtime.ota import in_rollout

from fleet_loop import dashboard
from fleet_loop.hub import Hub, HubConfig
from fleet_loop.loop import FleetConfig, bootstrap_policy, build_fleet, run_loop
from fleet_loop.sim import long_reach, miscalibrated, nominal, success_rate


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    """One loop run, shared across the tests that examine it."""
    config = FleetConfig(
        root=tmp_path_factory.mktemp("fleet"), rounds=8, episodes_per_round=60, seed=0
    )
    return run_loop(config)


def test_the_fleet_gets_better_from_its_own_data(completed):
    report, hub, nodes = completed
    assert report.initial_success_rate < 0.5
    assert report.final_success_rate > 0.85
    assert report.versions_published >= 1


def test_the_first_policy_is_good_only_where_it_was_trained():
    """The premise. Without this gap there is nothing for the loop to do."""
    policy, _ = bootstrap_policy(nominal(), seed=0)
    assert success_rate(policy, nominal(), 40) > 0.7
    assert success_rate(policy, miscalibrated(), 40) < 0.35
    assert success_rate(policy, long_reach(), 40) < 0.2


def test_every_release_is_traceable_to_the_data_that_produced_it(completed):
    _, hub, _ = completed
    retrained = [r for r in hub.release_log if r["dataset_hash"]]
    assert retrained, "no release recorded which dataset it came from"

    for record in retrained:
        assert record["train"]["n_samples"] > 0
        matching = [d for d in hub.datasets if d.dataset_hash == record["dataset_hash"]]
        assert matching, f"v{record['version']} names a dataset the hub does not have"
        assert matching[0].episodes_by_node


def test_a_release_reaches_a_cohort_before_the_fleet(completed):
    """Nothing goes to every node without being observed on some of them first."""
    _, hub, nodes = completed
    canaried = [r for r in hub.release_log if r["version"] > 1]
    assert canaried, "no release after the factory image"

    for record in canaried:
        if record["notes"].startswith("rollback"):
            continue
        cohort = [
            node.config.node_id
            for node in nodes
            if in_rollout(node.config.node_id, record["version"], record["rollout_percent"])
        ]
        assert 0 < len(cohort) < len(nodes) or record["rollout_percent"] == 100


def test_promotion_happens_on_fleet_evidence_not_on_a_timer(completed):
    report, _, _ = completed
    promotions = [r.canary for r in report.rounds if r.canary and r.canary.promoted]
    assert promotions, "nothing was promoted"
    for canary in promotions:
        assert canary.ci_low > -canary.tolerance
        assert len(canary.per_node) >= 2


def test_the_loop_closes_on_a_fraction_of_the_data(completed):
    """The claim triage is making, in the unit that matters.

    States, not episodes: a failed episode runs to the step limit and a
    successful one settles in a tenth of that, so counting episodes hides most
    of the bytes.
    """
    report, hub, _ = completed
    observed_states = sum(o.steps for o in hub.ingest.outcomes.values())
    shipped_states = sum(t.steps for t in hub.ingest.trajectories)

    assert shipped_states < observed_states * 0.3, "triage is not being selective"
    assert report.final_success_rate > 0.85, "and it still has to close the loop"


def test_every_episode_is_measured_even_though_most_are_not_collected(completed):
    """The measurement channel is not filtered by the collection heuristic.

    A gate whose input is selected by a value heuristic is measuring the
    heuristic.
    """
    report, hub, _ = completed
    observed_episodes = sum(
        stats["episodes"] for r in report.rounds for stats in r.per_node.values()
    )
    assert hub.ingest.stats()["outcomes"] == observed_episodes
    assert len(hub.ingest.trajectories) < observed_episodes / 3


def test_the_shipped_set_is_not_only_failures(completed):
    """A failure-only pipeline poisons the model that generates the next round.

    The success quota is what stops it, and this is the assertion that would
    fail if the reservation were removed.
    """
    _, hub, _ = completed
    successes = sum(t.success for t in hub.ingest.trajectories)
    assert successes > 0
    assert 0.05 < successes / len(hub.ingest.trajectories) < 0.95


def test_no_sync_exceeded_its_budget(completed):
    report, _, _ = completed
    for round_report in report.rounds:
        for node_id, stats in round_report.sync.items():
            assert stats["budget_used"] <= 1.0, f"{node_id} overran its budget"


def test_nothing_was_quarantined_in_a_healthy_run(completed):
    """The validator must not be rejecting the fleet's own well-formed data."""
    _, hub, _ = completed
    assert hub.ingest.stats()["quarantined"] == 0


def test_the_dashboard_renders_and_serialises(completed, tmp_path):
    _, hub, nodes = completed
    state = dashboard.fleet_state(hub, nodes)

    assert json.dumps(state)
    assert len(state["nodes"]) == 3
    assert state["published_version"] >= 1

    path = dashboard.write_html(state, tmp_path / "fleet.html")
    text = path.read_text()
    assert "fleet" in text
    for node in nodes:
        assert node.config.node_id in text


def test_version_skew_is_shown_rather_than_hidden(completed):
    """Skew is the normal state of a fleet, not an anomaly. A dashboard showing
    the published version and calling it the fleet's version shows intentions."""
    _, hub, nodes = completed
    state = dashboard.fleet_state(hub, nodes)
    assert state["versions_running"] == sorted({n.policy_version for n in nodes})
    assert state["version_skew"] == (len(state["versions_running"]) > 1)


# -- the release that passes every check and is worse ------------------------


def test_a_regression_the_hub_cannot_simulate_is_caught_by_the_fleet(tmp_path):
    """The whole reason the canary stage exists.

    The candidate is trained on the factory data alone — a plausible pipeline
    incident. It loads, it runs inside the latency budget, it passes the device
    health gate, and it scores as well as the incumbent in the hub's simulator,
    because the hub's simulator is the nominal robot. Only the nodes can tell.
    """
    config = FleetConfig(root=tmp_path / "fleet", rounds=6, episodes_per_round=60, seed=0)
    report, hub, nodes = run_loop(config)
    incumbent = hub.current_version
    assert incumbent > 1, (
        "the fleet never promoted a retrained release, so the factory image is "
        "still the incumbent and there is no regression to detect"
    )

    regressed, _ = bootstrap_policy(nominal(), seed=99)
    shippable, _, why = hub.worth_shipping(regressed)
    assert shippable, f"the hub's simulator saw the problem, so this test proves nothing: {why}"

    version = max(hub._known_versions()) + 1
    percent = hub.canary_percent_for([n.config.node_id for n in nodes], version)
    hub.publish(regressed, rollout_percent=percent, notes="trained without the fleet shards")

    for _ in range(2):
        for node in nodes:
            node.poll_updates()
            node.run_episodes(60)
            hub.report_outcomes(node.take_outcomes())

    canary = hub.evaluate_canary()
    assert canary.verdict == "ROLLBACK", canary.summary()
    assert canary.ci_high < -canary.tolerance

    restored = hub.roll_back(canary)
    assert restored > version, "versions must only ever go up"

    for node in nodes:
        node.poll_updates()
    assert {n.policy_version for n in nodes} == {restored}

    running = hub.policy_of(restored).content_hash()
    assert running == hub.policy_of(incumbent).content_hash(), (
        "the rollback did not restore the incumbent's policy"
    )


def test_a_hub_with_no_data_does_not_retrain(tmp_path):
    """Retraining on nothing is a full canary cycle spent to learn nothing."""
    hub = Hub(HubConfig(root=tmp_path / "hub"))
    should, reason = hub.should_retrain()
    assert not should
    assert "new episodes" in reason


def test_retraining_needs_new_coverage_not_just_new_bytes(tmp_path):
    config = FleetConfig(root=tmp_path / "fleet", rounds=2, episodes_per_round=40, seed=0)
    hub = Hub(HubConfig(root=tmp_path / "hub", retrain_min_coverage_gain=0.99))
    policy, _ = bootstrap_policy(nominal(), episodes=60, seed=0)
    hub.publish(policy, rollout_percent=100, notes="factory")

    nodes = build_fleet(config, hub)
    for node in nodes:
        node.install_local(hub.builds / "1")
        node.run_episodes(80)
        from fleet_loop.node import sync as sync_node

        sync_node(node, hub, 400_000)

    hub.retrain()  # first assembly: everything is new
    for node in nodes:
        node.run_episodes(10)
        from fleet_loop.node import sync as sync_node

        sync_node(node, hub, 4_000)

    should, reason = hub.should_retrain()
    assert not should
    assert "shards are new" in reason or "new episodes" in reason
