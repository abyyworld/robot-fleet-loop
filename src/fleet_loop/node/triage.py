"""Deciding, on the node, which episodes are worth sending.

The constraint people name first is bandwidth. The constraint that actually
binds is **labelling**: the states a policy struggles in are the states somebody
has to teleoperate a correction for, and a human's time is more expensive than a
cellular modem's. Triage is therefore not "compress the log" — it is "choose
which states are worth a person's attention", and the ranking is the product.

Four signals, scored per episode:

``failed``
    The policy did not settle on the goal. The highest-value states there are,
    because they are the ones the current policy provably cannot handle.
``near_miss``
    Succeeded, but slowly or barely. These are where the policy is about to
    fail, and they are invisible in a success-rate metric until they cross over.
``device_flagged``
    The on-device detectors fired — out-of-distribution observation, saturated
    action, inference failure. This is `edge-policy-runtime`'s up-link feeding the
    data channel rather than only the dashboard.
``novel``
    The episode visited a region of the goal space the accepted dataset has
    little of. Two hundred more episodes of the same failure teach nothing that
    the first twenty did not.

**And a quota of ordinary successes, deliberately.** This is the part that is
easy to leave out and expensive to leave out. If the fleet only ever uploads
failures, the training set drifts to contain nothing but hard states, the
retrained policy is optimised for a distribution the robot is not in, and each
round makes the skew worse — a feedback loop where the data selection rule
poisons the model that generates the next round of data. So a fixed fraction of
the byte budget is reserved for uniformly-sampled ordinary episodes. They score
low and would never win on value; the reservation is what gets them through, and
:func:`select_within_budget` enforces it before it ranks anything.

None of this touches :class:`~fleet_loop.wire.EpisodeOutcome`. Every episode
reports its outcome regardless of what triage decides — the measurement channel
must not be filtered by the collection heuristic, or the canary gate ends up
measuring the heuristic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..sim import MAX_STEPS
from ..wire import Trajectory

# Fraction of the byte budget reserved for ordinary, uninteresting episodes.
SUCCESS_QUOTA = 0.2

WEIGHTS = {
    "failed": 1.0,
    "near_miss": 0.45,
    "device_flagged": 0.35,
    "novel": 0.5,
}


@dataclass
class Coverage:
    """A cheap count of which regions of the goal space the node has shipped.

    Deliberately coarse — a bucket per joint per sign, not a density model. The
    question is "have we sent a lot of this already?", and answering it with
    something a node has to maintain and a reviewer has to trust is a worse
    trade than answering it approximately.
    """

    bins: int = 4
    counts: Counter = field(default_factory=Counter)

    def key(self, trajectory: Trajectory) -> tuple[int, ...]:
        # The goal is recoverable from the first observation: joints, then the
        # error to the goal.
        first = trajectory.observations[0]
        n = (len(first) - 4) // 2
        goal = first[:n] + first[n : 2 * n]
        return tuple(np.clip(((goal + 1.5) / 3.0 * self.bins).astype(int), 0, self.bins - 1))

    def novelty(self, trajectory: Trajectory) -> float:
        """1.0 for a region never sent, decaying as more of it is sent."""
        return 1.0 / (1.0 + self.counts[self.key(trajectory)])

    def record(self, trajectory: Trajectory) -> None:
        self.counts[self.key(trajectory)] += 1

    def distinct_regions(self) -> int:
        return len(self.counts)


@dataclass
class TriageScore:
    trajectory: Trajectory
    value: float
    reasons: dict[str, float]
    per_state_value: float = 0.0

    @property
    def nbytes(self) -> int:
        return self.trajectory.nbytes

    @property
    def value_per_byte(self) -> float:
        return self.value / max(self.nbytes, 1)

    def top_reason(self) -> str:
        if not self.reasons:
            return "quota"
        return max(self.reasons.items(), key=lambda kv: kv[1])[0]


def score(trajectory: Trajectory, coverage: Coverage) -> TriageScore:
    """Value one episode. Reasons are kept so the hub can audit the heuristic.

    **The weights are per state, and the episode's value is the weight times its
    length.** That is not a detail. The first version of this scored per episode
    and ranked by value per byte, which is a length penalty in disguise: an
    episode's size is proportional to its steps, so a 120-step failure had to be
    ten times more valuable than a 12-step success just to break even. It never
    was, and the result was a fleet succeeding 47% of the time shipping a
    training set that was 88% successes — the exact inversion of what triage is
    for, produced by a ranking rule that looked obviously correct.

    Counting per state fixes it for a concrete reason rather than by tuning: the
    thing being bought with bandwidth is *states worth labelling*, and a failure
    that took 120 steps to fail contains 120 of them. Value per byte then
    compares information density rather than episode length, which is what the
    knapsack wanted in the first place.
    """
    reasons: dict[str, float] = {}

    if not trajectory.success:
        reasons["failed"] = WEIGHTS["failed"]
    elif trajectory.steps > 0.6 * MAX_STEPS:
        # Succeeded, but used most of its budget getting there.
        reasons["near_miss"] = WEIGHTS["near_miss"]

    if trajectory.flags:
        reasons["device_flagged"] = WEIGHTS["device_flagged"] * min(len(trajectory.flags), 3) / 3

    novelty = coverage.novelty(trajectory)
    if novelty > 0.25:
        reasons["novel"] = WEIGHTS["novel"] * novelty

    per_state = sum(reasons.values())
    return TriageScore(
        trajectory,
        value=per_state * trajectory.steps,
        reasons=reasons,
        per_state_value=per_state,
    )


def select_within_budget(
    trajectories: list[Trajectory],
    coverage: Coverage,
    byte_budget: int,
    *,
    success_quota: float = SUCCESS_QUOTA,
    rng: np.random.Generator | None = None,
) -> tuple[list[TriageScore], dict]:
    """Choose what to ship under a hard byte budget.

    Greedy by value per byte. This is the fractional-knapsack relaxation applied
    to an integral problem, so it is not optimal — but the episodes are all
    within a factor of two in size, which is the regime where greedy is within a
    few percent, and an exact solver here would be complexity spent on the wrong
    part of the system.

    The success quota is filled *first*, from a uniform sample rather than from
    the top of the ranking. Filling it from the ranking would defeat the purpose:
    the highest-scoring successes are the near-misses, which are not ordinary
    episodes and do not correct the skew.
    """
    rng = rng or np.random.default_rng(0)
    scored = [score(t, coverage) for t in trajectories]

    selected: list[TriageScore] = []
    spent = 0
    taken: set[int] = set()

    # 1. The reservation for ordinary episodes.
    quota_bytes = int(byte_budget * success_quota)
    ordinary = [
        i for i, s in enumerate(scored) if s.trajectory.success and "near_miss" not in s.reasons
    ]
    rng.shuffle(ordinary)
    for i in ordinary:
        if spent + scored[i].nbytes > quota_bytes:
            break
        selected.append(scored[i])
        taken.add(i)
        spent += scored[i].nbytes

    quota_spent = spent

    # 2. The rest of the budget, by value density.
    ranked = sorted(
        (i for i in range(len(scored)) if i not in taken),
        key=lambda i: scored[i].value_per_byte,
        reverse=True,
    )
    for i in ranked:
        if scored[i].value <= 0:
            continue
        if spent + scored[i].nbytes > byte_budget:
            continue  # a smaller episode further down may still fit
        selected.append(scored[i])
        taken.add(i)
        spent += scored[i].nbytes

    for chosen in selected:
        coverage.record(chosen.trajectory)

    reasons: Counter = Counter()
    for chosen in selected:
        reasons[chosen.top_reason()] += 1

    stats = {
        "considered": len(trajectories),
        "selected": len(selected),
        "bytes_selected": spent,
        "byte_budget": byte_budget,
        "bytes_available": sum(t.nbytes for t in trajectories),
        "quota_bytes_used": quota_spent,
        "reasons": dict(reasons),
        "selected_success_rate": (
            sum(s.trajectory.success for s in selected) / len(selected) if selected else 0.0
        ),
        "observed_success_rate": (
            sum(t.success for t in trajectories) / len(trajectories) if trajectories else 0.0
        ),
    }
    return selected, stats
