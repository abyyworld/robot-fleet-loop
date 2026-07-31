"""The hub: ingest, decide whether to retrain, publish, canary, promote or undo.

Three decisions here are the ones a reviewer should look at.

**Retraining is triggered by new *coverage*, not by new bytes.** A thousand more
episodes of a failure already in the training set produce a checkpoint that has
to be evaluated, canaried and either promoted or rolled back — a full cycle of
fleet exposure — to learn nothing. The trigger asks for both a volume of new
episodes and a fraction of shards the previous dataset did not have.

**A candidate is checked in simulation before it costs any fleet exposure.**
The hub's model of a node is not the node, so this cannot decide promotion. It
can decide *not* to ship: a candidate that is worse than the incumbent in the
hub's own simulator will not be better on hardware, and putting it in front of
real robots to find that out is exposure spent for nothing.

**Rolling back publishes a new version containing the old policy.** Versions are
monotonic — the device's only question is "is this newer than what I run", and a
scheme where the answer can go backwards is a scheme where a device that missed
a poll can end up on the wrong side of a rollback forever. So undoing v4 means
publishing v5 whose payload is v3's policy, with the reason in the notes. The
device-local rollback in `edge-policy-runtime` is a different mechanism for a
different failure: that one is for a device that cannot run the release at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from edge_runtime import bundle as bundle_mod
from edge_runtime.bundle import HealthSpec
from edge_runtime.ota import publish as publish_release
from edge_runtime.policy import ACTION_COLUMNS, Policy, default_obs_columns

from ..sim import NodeEnvironment, ReachEnv, nominal, success_rate
from ..wire import EpisodeOutcome, ShardManifest, SyncPlan
from . import canary as canary_mod
from .dataset import DatasetVersion, TrainingSet, assemble, coverage_gain
from .ingest import Ingest, Validator
from .train import TrainConfig, TrainReport, train


@dataclass
class HubConfig:
    root: Path
    obs_columns: list[str] = field(default_factory=default_obs_columns)
    action_columns: list[str] = field(default_factory=lambda: list(ACTION_COLUMNS))

    retrain_min_new_episodes: int = 100
    retrain_min_coverage_gain: float = 0.25
    canary_percent: int = 34
    tolerance: float = 0.03
    min_episodes_per_arm: int = 40
    # What the hub can simulate: the nominal robot, as specified. It does not
    # have a model of node 2's worn joints or node 3's workspace — those are
    # precisely the things nobody wrote down, and they are why the fleet has to
    # be asked. This is the reason `worth_shipping` can veto a candidate but can
    # never approve one.
    reference_environments: list[NodeEnvironment] = field(default_factory=lambda: [nominal()])


class ReleaseRecord(dict):
    """A published version, as a plain dict so the log serialises trivially."""


class Hub:
    """The central half of the loop."""

    def __init__(self, config: HubConfig) -> None:
        self.config = config
        self.root = Path(config.root)
        self.releases = self.root / "releases"
        self.builds = self.root / "builds"
        self.root.mkdir(parents=True, exist_ok=True)

        self.ingest = Ingest(
            Validator(config.obs_columns, config.action_columns, known_versions=set())
        )
        self.release_log: list[ReleaseRecord] = []
        self.datasets: list[DatasetVersion] = []
        self.current_version = 0
        self.candidate_version: int | None = None
        self._episodes_at_last_train = 0

    # -- HubEndpoint ---------------------------------------------------------

    def report_outcomes(self, outcomes: list[EpisodeOutcome]) -> list[str]:
        return self.ingest.report_outcomes(outcomes)

    def offer(self, manifests: list[ShardManifest]) -> SyncPlan:
        accept, already = self.ingest.offer(manifests)
        return SyncPlan(accept=accept, already_have=already)

    def upload(self, manifest: ShardManifest, payload: bytes) -> bool:
        result = self.ingest.accept(manifest, payload)
        return result.ok or result.duplicate

    # -- releases ------------------------------------------------------------

    def _known_versions(self) -> set[int]:
        return {int(r["version"]) for r in self.release_log}

    def publish(
        self,
        policy: Policy,
        *,
        rollout_percent: int,
        notes: str,
        dataset: DatasetVersion | None = None,
        train_report: TrainReport | None = None,
        eval_success_rate: float | None = None,
    ) -> int:
        """Build, sign and publish a bundle. Returns the new version number."""
        version = max(self._known_versions(), default=0) + 1
        probe = self._probe_observations(policy)

        bundle_dir = bundle_mod.build(
            policy,
            version=version,
            out_dir=self.builds,
            probe=probe,
            health=HealthSpec(latency_budget_ms=20.0, max_action_deviation=1e-4),
            dataset_hash=dataset.dataset_hash if dataset else None,
            eval_success_rate=eval_success_rate,
            notes=notes,
        )
        pointer = publish_release(bundle_dir, self.releases, rollout_percent=rollout_percent)

        record = ReleaseRecord(
            version=version,
            published_at=datetime.now(timezone.utc).isoformat(),
            rollout_percent=rollout_percent,
            policy_id=pointer["policy_id"],
            dataset_hash=dataset.dataset_hash if dataset else None,
            train=train_report.as_dict() if train_report else None,
            sim_success_rate=eval_success_rate,
            state="canary" if rollout_percent < 100 else "stable",
            notes=notes,
        )
        self.release_log.append(record)
        # The validator only trusts shards from versions this hub published.
        self.ingest.validator.known_versions = self._known_versions()

        if rollout_percent >= 100:
            self.current_version = version
            self.candidate_version = None
        else:
            self.candidate_version = version
        return version

    def _probe_observations(self, policy: Policy, n: int = 96) -> np.ndarray:
        """States the policy actually visits, for the device's health gate."""
        env = ReachEnv(env=self.config.reference_environments[0])
        observations = []
        obs = env.reset(seed=1234)
        for _ in range(n):
            observations.append(obs)
            obs, done, _ = env.step(policy.forward(obs))
            if done:
                obs = env.reset()
        return np.asarray(observations, dtype=np.float32)

    def policy_of(self, version: int) -> Policy:
        return bundle_mod.load_policy(self.builds / str(version))

    def canary_percent_for(self, node_ids: list[str], version: int, minimum_nodes: int = 2) -> int:
        """The smallest rollout percentage that puts `minimum_nodes` in the cohort.

        A three-node fleet cannot run a ten-percent canary. The smallest cohort
        that supports a within-node before/after comparison on two nodes is
        two-thirds of it, and the cohort is chosen by a salted hash the hub does
        not control — so the hub asks what percentage actually covers enough
        nodes rather than picking a number that sounds cautious and produces an
        `INCONCLUSIVE` verdict forever.

        The tension between limiting exposure and having the power to detect a
        regression is real and does not go away with fleet size. It only gets
        cheaper: on five hundred nodes the same two-node minimum is 0.4%.
        """
        from edge_runtime.ota import in_rollout

        for percent in range(1, 101):
            covered = sum(in_rollout(node_id, version, percent) for node_id in node_ids)
            if covered >= minimum_nodes and covered < len(node_ids):
                return percent
        # Everything is in the cohort at that point, which means there is no
        # control arm — the caller should treat this as "too small to canary".
        return 100

    # -- retraining ----------------------------------------------------------

    def new_episodes(self) -> int:
        return len(self.ingest.trajectories) - self._episodes_at_last_train

    def should_retrain(self) -> tuple[bool, str]:
        """Both a volume and a novelty condition, and the reason either way."""
        new = self.new_episodes()
        if new < self.config.retrain_min_new_episodes:
            return False, (
                f"{new} new episodes since the last training set; "
                f"need {self.config.retrain_min_new_episodes}"
            )

        candidate = assemble(self.ingest.trajectories, list(self.ingest.shards))
        gain = coverage_gain(candidate.version, self.datasets[-1] if self.datasets else None)
        if gain < self.config.retrain_min_coverage_gain:
            return False, (
                f"{new} new episodes, but only {gain:.0%} of shards are new; "
                f"need {self.config.retrain_min_coverage_gain:.0%} — retraining on data "
                "the last run already saw costs a full canary cycle to learn nothing"
            )
        return True, f"{new} new episodes, {gain:.0%} of shards unseen"

    def assemble_dataset(self) -> TrainingSet:
        training_set = assemble(
            self.ingest.trajectories,
            list(self.ingest.shards),
            parent=self.datasets[-1].dataset_hash if self.datasets else None,
        )
        self.datasets.append(training_set.version)
        training_set.version.write(
            self.root / "datasets" / f"{training_set.version.dataset_hash[:12]}.json"
        )
        self._episodes_at_last_train = len(self.ingest.trajectories)
        return training_set

    def retrain(self, config: TrainConfig | None = None) -> tuple[Policy, TrainingSet, TrainReport]:
        training_set = self.assemble_dataset()
        policy, report = train(
            training_set.observations,
            training_set.actions,
            self.config.obs_columns,
            config=config,
            name=f"fleet-bc-{training_set.version.dataset_hash[:8]}",
        )
        return policy, training_set, report

    # -- the pre-flight check ------------------------------------------------

    def simulate(self, policy: Policy, episodes: int = 60) -> dict[str, float]:
        """Success rate per reference environment. Cheap, and not the decision."""
        return {
            environment.name: success_rate(policy, environment, episodes=episodes, seed=17)
            for environment in self.config.reference_environments
        }

    def worth_shipping(self, policy: Policy, episodes: int = 60) -> tuple[bool, dict, str]:
        """Refuse to spend fleet exposure on a candidate that is already worse."""
        rates = self.simulate(policy, episodes=episodes)
        mean = float(np.mean(list(rates.values())))
        if self.current_version == 0:
            return True, rates, f"first release, {mean:.0%} in simulation"

        incumbent = self.simulate(self.policy_of(self.current_version), episodes=episodes)
        incumbent_mean = float(np.mean(list(incumbent.values())))
        if mean + self.config.tolerance < incumbent_mean:
            return (
                False,
                rates,
                f"{mean:.0%} vs the incumbent's {incumbent_mean:.0%} in simulation — "
                "a candidate that is worse in the hub's own model will not be better "
                "on hardware, and finding that out on real robots is exposure spent "
                "for nothing",
            )
        return True, rates, f"{mean:.0%} vs the incumbent's {incumbent_mean:.0%} in simulation"

    # -- the canary decision -------------------------------------------------

    def evaluate_canary(self) -> canary_mod.CanaryReport | None:
        """Compare the canary version against the incumbent on fleet outcomes."""
        if self.candidate_version is None:
            return None
        return canary_mod.evaluate(
            list(self.ingest.outcomes.values()),
            candidate_version=self.candidate_version,
            incumbent_version=self.current_version,
            tolerance=self.config.tolerance,
            min_episodes_per_arm=self.config.min_episodes_per_arm,
        )

    def promote(self, report: canary_mod.CanaryReport) -> int:
        """Take the canary to the whole fleet."""
        version = report.candidate_version
        publish_release(self.builds / str(version), self.releases, rollout_percent=100)
        self._set_state(version, "stable")
        self.current_version = version
        self.candidate_version = None
        return version

    def roll_back(self, report: canary_mod.CanaryReport) -> int:
        """Undo a canary by publishing the incumbent's policy as a *new* version."""
        failed = report.candidate_version
        incumbent = self.policy_of(self.current_version)
        version = self.publish(
            incumbent,
            rollout_percent=100,
            notes=(
                f"rollback of v{failed}: {report.delta:+.1%} success "
                f"[{report.ci_low:+.1%}, {report.ci_high:+.1%}] on the fleet"
            ),
        )
        self._set_state(failed, "rolled_back")
        self.candidate_version = None
        return version

    def _set_state(self, version: int, state: str) -> None:
        for record in self.release_log:
            if record["version"] == version:
                record["state"] = state

    # -- reporting -----------------------------------------------------------

    def status(self) -> dict:
        return {
            "current_version": self.current_version,
            "candidate_version": self.candidate_version,
            "releases": len(self.release_log),
            "datasets": len(self.datasets),
            "ingest": self.ingest.stats(),
            "new_episodes_since_training": self.new_episodes(),
        }

    def write_release_log(self) -> Path:
        path = self.root / "releases.json"
        path.write_text(json.dumps(self.release_log, indent=2))
        return path
