"""Tests for the decision the device health gate cannot make."""

from __future__ import annotations

from conftest import make_outcomes
from fleet_loop.hub.canary import evaluate, required_episodes


def fleet(candidate: dict[str, float], incumbent: dict[str, float], n: int = 120):
    outcomes = []
    for node, rate in incumbent.items():
        outcomes += make_outcomes(node, 1, n, rate)
    for node, rate in candidate.items():
        outcomes += make_outcomes(node, 2, n, rate)
    return outcomes


def run(candidate, incumbent, n=120, **kwargs):
    return evaluate(
        fleet(candidate, incumbent, n),
        candidate_version=2,
        incumbent_version=1,
        **kwargs,
    )


# -- the three verdicts ------------------------------------------------------


def test_a_clear_improvement_is_promoted():
    report = run({"a": 0.90, "b": 0.88}, {"a": 0.60, "b": 0.58})
    assert report.verdict == "PROMOTE"
    assert report.delta > 0.25
    assert report.ci_low > 0


def test_a_clear_regression_is_rolled_back():
    report = run({"a": 0.40, "b": 0.38}, {"a": 0.85, "b": 0.83})
    assert report.verdict == "ROLLBACK"
    assert report.ci_high < -0.03


def test_a_difference_the_run_cannot_resolve_is_inconclusive():
    """Absence of evidence is not evidence of absence.

    A comparison whose interval is wider than the tolerance it is testing has
    not shown a release is safe; it has shown the study was too small.
    """
    report = run({"a": 0.70, "b": 0.68}, {"a": 0.72, "b": 0.70}, n=45)
    assert report.verdict == "INCONCLUSIVE"
    assert report.required_episodes_per_arm > 45
    assert "wider than" in report.reasons[0]


def test_an_identical_policy_needs_the_study_the_power_calculation_asks_for():
    """The gate asks "is this a regression beyond tolerance", not "is this
    better" — so an identical policy should pass. But only once the study is
    large enough to say so, and the two halves of this test agree with each
    other: the episode count `required_episodes` demands is the count at which
    the verdict actually flips.
    """
    small = run({"a": 0.75, "b": 0.75}, {"a": 0.75, "b": 0.75}, n=400)
    assert small.verdict == "INCONCLUSIVE"
    assert small.required_episodes_per_arm > 2500

    large = run({"a": 0.75, "b": 0.75}, {"a": 0.75, "b": 0.75}, n=4000)
    assert large.verdict == "PROMOTE"


# -- what makes the comparison trustworthy -----------------------------------


def test_only_nodes_that_ran_both_versions_count():
    """Comparing canary nodes against different non-canary nodes is an
    observational study across different hardware, and with a handful of nodes
    the hardware difference dwarfs the policy difference."""
    outcomes = make_outcomes("canary", 2, 120, 0.9) + make_outcomes("control", 1, 120, 0.5)
    report = evaluate(outcomes, candidate_version=2, incumbent_version=1)

    assert report.verdict == "INCONCLUSIVE"
    assert "ran both versions" in report.reasons[0]


def test_a_single_node_is_not_a_fleet():
    outcomes = make_outcomes("a", 2, 200, 0.9) + make_outcomes("a", 1, 200, 0.5)
    report = evaluate(outcomes, candidate_version=2, incumbent_version=1, min_nodes=2)
    assert report.verdict == "INCONCLUSIVE"


def test_a_node_with_too_few_episodes_is_excluded_not_averaged_in():
    outcomes = fleet({"a": 0.9, "b": 0.9}, {"a": 0.5, "b": 0.5}, n=120)
    outcomes += make_outcomes("c", 2, 3, 0.0) + make_outcomes("c", 1, 3, 1.0)
    report = evaluate(outcomes, candidate_version=2, incumbent_version=1)

    assert report.verdict == "PROMOTE", "three episodes from node c swung the verdict"
    assert "c" in report.per_node  # still reported, just not counted


def test_a_release_that_collapses_on_one_node_is_refused():
    """The failure aggregates hide. On a heterogeneous fleet it is the likely
    shape of a bad release: fine everywhere except the node that is different.
    """
    report = run({"a": 0.98, "b": 0.97, "c": 0.30}, {"a": 0.70, "b": 0.70, "c": 0.72})
    assert report.verdict == "ROLLBACK"
    assert "collapsed on c" in report.reasons[-1]


def test_nodes_are_weighted_equally_not_by_episode_count():
    """A node that happens to have run ten times more episodes should not decide
    the fleet's verdict."""
    outcomes = make_outcomes("busy", 1, 1000, 0.80) + make_outcomes("busy", 2, 1000, 0.82)
    outcomes += make_outcomes("quiet", 1, 100, 0.80) + make_outcomes("quiet", 2, 100, 0.20)
    report = evaluate(outcomes, candidate_version=2, incumbent_version=1)

    assert report.verdict == "ROLLBACK"
    assert report.delta < -0.2, "the busy node drowned out the collapse on the quiet one"


def test_per_node_detail_is_always_reported():
    report = run({"a": 0.9, "b": 0.5}, {"a": 0.6, "b": 0.6})
    assert set(report.per_node) == {"a", "b"}
    assert report.per_node["a"]["n_candidate"] == 120
    assert report.per_node["b"]["delta"] < 0


def test_the_report_serialises():
    """A dashboard and a release log both have to be able to store it."""
    import json

    assert json.dumps(run({"a": 0.9, "b": 0.9}, {"a": 0.6, "b": 0.6}).as_dict())


# -- power -------------------------------------------------------------------


def test_required_episodes_grows_steeply_as_the_effect_shrinks():
    assert required_episodes(0.7, 0.05) > 4 * required_episodes(0.7, 0.10)
    # Resolving three points around a 70% baseline is a study most fleets will
    # not run by accident.
    assert required_episodes(0.7, 0.03) > 3000


def test_required_episodes_is_zero_for_a_meaningless_effect():
    assert required_episodes(0.7, 0.0) == 0
