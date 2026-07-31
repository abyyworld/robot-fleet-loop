"""The loop itself, driven end to end.

One round is: run, triage, sync, validate, maybe retrain, maybe canary, decide.
Everything each half needs from the other crosses the interfaces in
:mod:`fleet_loop.wire` — episode outcomes and shards going up, signed bundles
coming down — and the loop below is the scheduler, not the logic.

The fleet is deliberately heterogeneous. One well-calibrated node, one with a
constant joint bias standing in for a worn arm, one working further out than the
training distribution went. A loop demonstrated on identical nodes demonstrates
nothing: the first policy would already be good everywhere and there would be
nothing for the fleet to teach it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from edge_runtime.ota import DirectoryReleaseSource

from .hub import Hub, HubConfig
from .hub.canary import CanaryReport
from .hub.train import TrainConfig
from .node import BandwidthEstimate, FleetNode, NodeConfig, sync
from .sim import ExpertPolicy, NodeEnvironment, ReachEnv, long_reach, miscalibrated, nominal


@dataclass
class FleetConfig:
    root: Path
    rounds: int = 8
    episodes_per_round: int = 60
    # Per node, per round. About a quarter of what a round of episodes weighs on
    # the wire, which is what makes triage a decision rather than a formality: a
    # budget that fits everything is not a budget.
    wire_budget_bytes: int = 40_000
    seed: int = 0
    environments: list[NodeEnvironment] = field(
        default_factory=lambda: [nominal(), miscalibrated(), long_reach()]
    )


@dataclass
class RoundReport:
    round: int
    per_node: dict[str, dict] = field(default_factory=dict)
    sync: dict[str, dict] = field(default_factory=dict)
    retrained: bool = False
    retrain_reason: str = ""
    published_version: int | None = None
    canary: CanaryReport | None = None
    action: str = ""
    hub: dict = field(default_factory=dict)

    @property
    def fleet_success_rate(self) -> float:
        rates = [v["success_rate"] for v in self.per_node.values()]
        return float(np.mean(rates)) if rates else 0.0


@dataclass
class LoopReport:
    rounds: list[RoundReport] = field(default_factory=list)
    initial_success_rate: float = 0.0
    final_success_rate: float = 0.0
    versions_published: int = 0
    rollbacks: int = 0

    def as_dict(self) -> dict:
        return {
            "rounds": len(self.rounds),
            "initial_success_rate": round(self.initial_success_rate, 4),
            "final_success_rate": round(self.final_success_rate, 4),
            "versions_published": self.versions_published,
            "rollbacks": self.rollbacks,
            "per_round": [
                {
                    "round": r.round,
                    "fleet_success_rate": round(r.fleet_success_rate, 4),
                    "action": r.action,
                    "published_version": r.published_version,
                    "canary": r.canary.as_dict() if r.canary else None,
                }
                for r in self.rounds
            ],
        }


# -- construction ------------------------------------------------------------


def bootstrap_policy(environment: NodeEnvironment, episodes: int = 250, seed: int = 0):
    """The v1 that ships from the factory: expert data from *one* environment.

    This is the honest version of "we trained a policy before deployment". It is
    good on the conditions it saw and mediocre elsewhere, which is exactly the
    situation a fleet loop exists to fix — and it is why the fleet's own data has
    somewhere to go.
    """
    from .hub.train import train

    env = ReachEnv(env=environment)
    expert = ExpertPolicy()
    observations, actions = [], []
    for episode in range(episodes):
        obs = env.reset(seed=seed * 1000 + episode)
        while True:
            observations.append(obs)
            obs, done, _ = env.step(expert.forward(obs))
            actions.append(env.commanded)
            if done:
                break

    policy, report = train(
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        env.obs_columns,
        config=TrainConfig(seed=seed),
        name="factory-v1",
    )
    return policy, report


def build_fleet(config: FleetConfig, hub: Hub) -> list[FleetNode]:
    source = DirectoryReleaseSource(hub.releases)
    return [
        FleetNode(
            NodeConfig(
                node_id=f"node-{i + 1:02d}-{environment.name}",
                root=Path(config.root) / "nodes" / f"node-{i + 1:02d}",
                environment=environment,
            ),
            source=source,
        )
        for i, environment in enumerate(config.environments)
    ]


def build_hub(config: FleetConfig) -> Hub:
    # The hub gets a simulator of the *nominal* robot only — the one that was
    # specified. What each node's arm has actually become is what the fleet is
    # there to report.
    return Hub(HubConfig(root=Path(config.root) / "hub"))


# -- the loop ----------------------------------------------------------------


def run_loop(config: FleetConfig, on_round=None) -> tuple[LoopReport, Hub, list[FleetNode]]:
    """Run the whole thing. `on_round` is called with each :class:`RoundReport`."""
    rng = np.random.default_rng(config.seed)
    hub = build_hub(config)

    # v1: trained before deployment, on the nominal node's conditions only.
    factory, factory_report = bootstrap_policy(config.environments[0], seed=config.seed)
    rates = hub.simulate(factory)
    hub.publish(
        factory,
        rollout_percent=100,
        notes="factory image, trained on nominal conditions only",
        train_report=factory_report,
        eval_success_rate=float(np.mean(list(rates.values()))),
    )

    nodes = build_fleet(config, hub)
    for node in nodes:
        node.install_local(hub.builds / "1")

    estimates = {node.config.node_id: BandwidthEstimate() for node in nodes}
    report = LoopReport()

    for round_index in range(1, config.rounds + 1):
        round_report = RoundReport(round=round_index)

        # 1. Nodes take whatever release applies to them, then work.
        for node in nodes:
            node.poll_updates()
            round_report.per_node[node.config.node_id] = node.run_episodes(
                config.episodes_per_round
            )

        # 2. Nodes report outcomes always, and ship what fits.
        for node in nodes:
            result = sync(
                node,
                hub,
                config.wire_budget_bytes,
                estimate=estimates[node.config.node_id],
                rng=rng,
            )
            round_report.sync[node.config.node_id] = {
                "episodes_sent": result.episodes_sent,
                "wire_bytes": result.wire_bytes,
                "budget_used": round(result.budget_used, 3),
                "duplicate_shards": result.duplicate_shards,
                "triage": result.triage,
            }

        # 3. The canary decision comes before retraining: a candidate that is
        #    still being judged must not be replaced by another one.
        round_report.canary = hub.evaluate_canary()
        if round_report.canary is not None:
            round_report.action = _act_on_canary(hub, round_report.canary, report)
        else:
            # 4. Retrain, if there is a reason to.
            round_report.action = _maybe_retrain(hub, nodes, round_report, report)

        round_report.hub = hub.status()
        report.rounds.append(round_report)
        if on_round:
            on_round(round_report)

    report.initial_success_rate = report.rounds[0].fleet_success_rate
    report.final_success_rate = report.rounds[-1].fleet_success_rate
    hub.write_release_log()
    return report, hub, nodes


def _maybe_retrain(
    hub: Hub, nodes: list[FleetNode], round_report: RoundReport, report: LoopReport
) -> str:
    should, reason = hub.should_retrain()
    round_report.retrain_reason = reason
    if not should:
        return f"no retrain — {reason}"

    round_report.retrained = True
    policy, training_set, train_report = hub.retrain()

    shippable, rates, why = hub.worth_shipping(policy)
    if not shippable:
        return f"candidate withheld — {why}"

    next_version = max(hub._known_versions(), default=0) + 1
    percent = hub.canary_percent_for([n.config.node_id for n in nodes], next_version)
    published = hub.publish(
        policy,
        rollout_percent=percent,
        notes=f"retrained on {training_set.version.n_episodes} fleet episodes",
        dataset=training_set.version,
        train_report=train_report,
        eval_success_rate=float(np.mean(list(rates.values()))),
    )
    report.versions_published += 1
    round_report.published_version = published
    return f"published v{published} to {percent}% of the fleet — {why}"


def _act_on_canary(hub: Hub, canary: CanaryReport, report: LoopReport) -> str:
    if canary.verdict == "PROMOTE":
        version = hub.promote(canary)
        return f"promoted v{version} to the whole fleet — {canary.summary()}"
    if canary.verdict == "ROLLBACK":
        version = hub.roll_back(canary)
        report.rollbacks += 1
        report.versions_published += 1
        return (
            f"rolled back, published v{version} carrying the previous policy — {canary.summary()}"
        )
    return f"canary still running — {canary.summary()}"
