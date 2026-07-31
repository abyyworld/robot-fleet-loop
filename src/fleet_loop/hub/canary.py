"""Deciding, from what the fleet reports, whether a release is better or worse.

The device health gate in `edge-policy-runtime` catches broken, incompatible and
slow. It cannot catch *runs fine, succeeds less often*, because answering that
needs episodes and a comparison against the incumbent. This is where that
happens.

**The comparison is within-node, before versus after.** Comparing canary nodes
against non-canary nodes is an observational study across different hardware in
different conditions, and with a handful of nodes the difference between them
dwarfs the difference between two policy versions. Restricting to nodes that ran
both versions removes node identity as a confounder entirely, which is the
dominant one here.

**The residual confound is time, and it is not removed.** A within-node
before/after comparison attributes to the release anything else that changed
between the two windows — wear, ambient conditions, a different operator. The
mitigation is that canary and non-canary nodes are observed over the same wall
clock window, so a fleet-wide temporal effect appears in both arms and can be
seen; it is a mitigation, not a fix, and a fleet of three nodes cannot do better
than that. Naming it is the honest option.

**Three verdicts, not two.** `INCONCLUSIVE` is the one most rollout systems
lack. A comparison whose interval is wider than the tolerance it is testing has
not shown the release is safe — it has shown the study was too small. Promoting
on it is a coin flip wearing a decision's clothing. The report says how many
episodes per arm would be needed instead.

**These statistics are weaker than an offline comparison, unavoidably.** An
offline harness evaluates two policies on *matched seeds* — the same episodes,
in the same order — which lets it pair the outcomes and remove episode difficulty
from the comparison entirely. Matched seeds do not exist in a fleet: two nodes
never face the same episode, and one node never faces the same episode twice.
So this is unpaired, it needs far more episodes for the same resolving power,
and :func:`required_episodes` is here to say how many rather than let anyone
assume.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..wire import EpisodeOutcome

DEFAULT_TOLERANCE = 0.03  # allowed regression in success rate
MIN_EPISODES_PER_ARM = 40
MIN_NODES = 2


@dataclass
class ArmSummary:
    n: int
    successes: int

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else float("nan")


@dataclass
class CanaryReport:
    verdict: str  # PROMOTE | ROLLBACK | INCONCLUSIVE
    candidate_version: int
    incumbent_version: int
    delta: float
    ci_low: float
    ci_high: float
    tolerance: float
    per_node: dict[str, dict] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    required_episodes_per_arm: int | None = None

    @property
    def promoted(self) -> bool:
        return self.verdict == "PROMOTE"

    def summary(self) -> str:
        head = (
            f"v{self.candidate_version} vs v{self.incumbent_version}: "
            f"{self.delta:+.1%} [{self.ci_low:+.1%}, {self.ci_high:+.1%}] "
            f"→ {self.verdict}"
        )
        return head + ("  — " + "; ".join(self.reasons) if self.reasons else "")

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "candidate_version": self.candidate_version,
            "incumbent_version": self.incumbent_version,
            "delta": round(self.delta, 5),
            "ci_low": round(self.ci_low, 5),
            "ci_high": round(self.ci_high, 5),
            "tolerance": self.tolerance,
            "per_node": self.per_node,
            "reasons": self.reasons,
            "required_episodes_per_arm": self.required_episodes_per_arm,
        }


def required_episodes(baseline_rate: float, effect: float, power: float = 0.8) -> int:
    """Episodes per arm needed to resolve `effect`, two-sided at alpha = 0.05.

    Normal approximation on two proportions. Approximate, and the point of
    reporting it is the order of magnitude: resolving a 3-point difference
    around a 70% baseline is a study most fleets will not run by accident.
    """
    if effect <= 0:
        return 0
    p1 = min(max(baseline_rate, 1e-6), 1 - 1e-6)
    p2 = min(max(p1 + effect, 1e-6), 1 - 1e-6)
    z_alpha, z_beta = 1.959964, {0.8: 0.8416, 0.9: 1.2816}.get(power, 0.8416)
    pooled = (p1 + p2) / 2
    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return int(math.ceil(numerator / (effect**2)))


def _split_by_node(
    outcomes: list[EpisodeOutcome], candidate: int, incumbent: int
) -> dict[str, tuple[list[bool], list[bool]]]:
    per_node: dict[str, tuple[list[bool], list[bool]]] = defaultdict(lambda: ([], []))
    for outcome in outcomes:
        if outcome.policy_version == candidate:
            per_node[outcome.node_id][0].append(bool(outcome.success))
        elif outcome.policy_version == incumbent:
            per_node[outcome.node_id][1].append(bool(outcome.success))
    return dict(per_node)


def evaluate(
    outcomes: list[EpisodeOutcome],
    *,
    candidate_version: int,
    incumbent_version: int,
    tolerance: float = DEFAULT_TOLERANCE,
    min_episodes_per_arm: int = MIN_EPISODES_PER_ARM,
    min_nodes: int = MIN_NODES,
    n_bootstrap: int = 4000,
    seed: int = 0,
) -> CanaryReport:
    """Compare two policy versions on what the fleet actually reported."""
    per_node_raw = _split_by_node(outcomes, candidate_version, incumbent_version)
    usable = {
        node: (cand, inc)
        for node, (cand, inc) in per_node_raw.items()
        if len(cand) >= min_episodes_per_arm and len(inc) >= min_episodes_per_arm
    }

    per_node = {
        node: {
            "candidate": ArmSummary(len(cand), sum(cand)).rate,
            "incumbent": ArmSummary(len(inc), sum(inc)).rate,
            "n_candidate": len(cand),
            "n_incumbent": len(inc),
            "delta": ArmSummary(len(cand), sum(cand)).rate - ArmSummary(len(inc), sum(inc)).rate,
        }
        for node, (cand, inc) in per_node_raw.items()
    }

    if len(usable) < min_nodes:
        short = [
            f"{node}: {len(c)} candidate / {len(i)} incumbent"
            for node, (c, i) in per_node_raw.items()
        ]
        return CanaryReport(
            verdict="INCONCLUSIVE",
            candidate_version=candidate_version,
            incumbent_version=incumbent_version,
            delta=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            tolerance=tolerance,
            per_node=per_node,
            reasons=[
                f"only {len(usable)} node(s) ran both versions with at least "
                f"{min_episodes_per_arm} episodes per arm; need {min_nodes}"
                + (f" ({'; '.join(short)})" if short else "")
            ],
            required_episodes_per_arm=min_episodes_per_arm,
        )

    delta, low, high = _stratified_bootstrap(usable, n_bootstrap=n_bootstrap, seed=seed)

    reasons: list[str] = []
    if low > -tolerance:
        verdict = "PROMOTE"
    elif high < -tolerance:
        verdict = "ROLLBACK"
        reasons.append(f"the interval rules out a regression smaller than {tolerance:.0%}")
    else:
        verdict = "INCONCLUSIVE"
        reasons.append(f"the interval is wider than the {tolerance:.0%} tolerance it is testing")

    # Per-task collapse, fleet edition: a candidate can hold its overall rate
    # while falling apart on one node. That is the failure aggregates hide, and
    # on a heterogeneous fleet it is the likely shape of a bad release.
    collapsed = [
        node
        for node, (cand, inc) in usable.items()
        if (sum(cand) / len(cand)) - (sum(inc) / len(inc)) < -3 * tolerance
    ]
    if collapsed and verdict == "PROMOTE":
        verdict = "ROLLBACK"
        reasons.append(f"held its overall rate but collapsed on {', '.join(sorted(collapsed))}")

    baseline = float(np.mean([sum(inc) / len(inc) for _, inc in usable.values()]))
    return CanaryReport(
        verdict=verdict,
        candidate_version=candidate_version,
        incumbent_version=incumbent_version,
        delta=delta,
        ci_low=low,
        ci_high=high,
        tolerance=tolerance,
        per_node=per_node,
        reasons=reasons,
        required_episodes_per_arm=(
            required_episodes(baseline, tolerance) if verdict == "INCONCLUSIVE" else None
        ),
    )


def _stratified_bootstrap(
    usable: dict[str, tuple[list[bool], list[bool]]],
    *,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Resample within each node, then average the per-node differences.

    Nodes are weighted equally rather than by episode count. A node that happens
    to have run ten times more episodes should not decide the fleet's verdict,
    and on a heterogeneous fleet the equal weighting is also the one that asks
    "does this work everywhere" rather than "does this work on average".
    """
    rng = np.random.default_rng(seed)
    arrays = {
        node: (np.asarray(cand, dtype=bool), np.asarray(inc, dtype=bool))
        for node, (cand, inc) in usable.items()
    }

    point = float(np.mean([c.mean() - i.mean() for c, i in arrays.values()]))

    samples = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        deltas = []
        for cand, inc in arrays.values():
            c = cand[rng.integers(0, len(cand), len(cand))]
            i = inc[rng.integers(0, len(inc), len(inc))]
            deltas.append(c.mean() - i.mean())
        samples[b] = float(np.mean(deltas))

    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(low), float(high)
