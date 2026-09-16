"""Command-line interface."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, dashboard
from .hub.canary import required_episodes
from .loop import FleetConfig, RoundReport, bootstrap_policy, run_loop
from .sim import ExpertPolicy, ZeroPolicy, long_reach, miscalibrated, nominal, success_rate

app = typer.Typer(
    add_completion=False,
    help="A closed fleet loop: local triage, selective sync, retrain, canary, roll back.",
    no_args_is_help=True,
)
console = Console()

RootOpt = typer.Option("var/fleet", "--root", help="Where fleet state is written.")


@app.command()
def version() -> None:
    console.print(f"robot-fleet-loop {__version__}")


@app.command()
def anchors(episodes: int = typer.Option(60)) -> None:
    """Confirm the task discriminates before believing any number from it.

    A do-nothing policy must fail and the expert must succeed, on every node's
    conditions. If these stop holding, every success rate this repo reports is
    measuring the environment rather than the policy.
    """
    environments = [nominal(), miscalibrated(), long_reach()]
    table = Table(title="task anchors")
    table.add_column("conditions")
    table.add_column("zero", justify="right")
    table.add_column("expert", justify="right")

    problems = []
    for environment in environments:
        zero = success_rate(ZeroPolicy(), environment, episodes=episodes)
        expert = success_rate(ExpertPolicy(), environment, episodes=episodes)
        table.add_row(
            environment.name,
            f"[red]{zero:.1%}[/]" if zero > 0.02 else f"{zero:.1%}",
            f"[red]{expert:.1%}[/]" if expert < 0.9 else f"{expert:.1%}",
        )
        if zero > 0.02:
            problems.append(f"{environment.name}: a do-nothing policy scores {zero:.1%}")
        if expert < 0.9:
            problems.append(f"{environment.name}: the expert only scores {expert:.1%}")

    console.print(table)
    if problems:
        for problem in problems:
            console.print(f"[red]{problem}[/]")
        raise typer.Exit(code=1)
    console.print("[green]the task discriminates[/] — trivial policies fail, the expert succeeds")


@app.command()
def power(
    baseline: float = typer.Option(0.7, help="Incumbent success rate."),
    tolerance: float = typer.Option(0.03, help="Regression the gate must detect."),
) -> None:
    """How many episodes per arm a fleet comparison actually needs."""
    table = Table(title=f"episodes per arm to resolve a difference around {baseline:.0%}")
    table.add_column("effect", justify="right")
    table.add_column("episodes per arm", justify="right")
    for effect in (0.20, 0.10, 0.05, tolerance, 0.02):
        table.add_row(f"{effect:.0%}", f"{required_episodes(baseline, effect):,}")
    console.print(table)
    console.print(
        "Unpaired, because two nodes never face the same episode. A paired "
        "offline comparison on matched seeds needs far fewer episodes for the "
        "same resolving power, and matched seeds do not exist in a fleet."
    )


@app.command()
def run(
    root: str = RootOpt,
    rounds: int = typer.Option(8, help="Loop iterations."),
    episodes: int = typer.Option(60, help="Episodes per node per round."),
    budget: int = typer.Option(40_000, help="Wire bytes per node per sync window."),
    fresh: bool = typer.Option(True, help="Clear existing fleet state first."),
    html_out: str = typer.Option("reports/fleet.html", "--html", help="Dashboard output."),
) -> None:
    """Run the loop end to end and show what happened each round."""
    if fresh:
        shutil.rmtree(root, ignore_errors=True)

    def show(report: RoundReport) -> None:
        console.rule(f"[bold]round {report.round}[/]  fleet {report.fleet_success_rate:.0%}")
        for node_id, stats in report.per_node.items():
            sync_stats = report.sync.get(node_id, {})
            triage = sync_stats.get("triage", {})
            console.print(
                f"[dim]│[/] {node_id:24s} v{stats['policy_version']}  "
                f"{stats['success_rate']:5.0%}  "
                f"sent {sync_stats.get('episodes_sent', 0):2d}/{triage.get('considered', 0):3d} "
                f"({sync_stats.get('wire_bytes', 0) / 1024:5.0f} KiB, "
                f"{sync_stats.get('budget_used', 0):.0%} of budget)  "
                f"{_reasons(triage)}"
            )
        console.print(f"[dim]│[/] [bold]{report.action}[/]")

    config = FleetConfig(
        root=Path(root), rounds=rounds, episodes_per_round=episodes, wire_budget_bytes=budget
    )
    report, hub, nodes = run_loop(config, on_round=show)

    console.rule("[bold]fleet")
    state = dashboard.render(hub, nodes, console=console)
    path = dashboard.write_html(state, html_out)

    console.rule("[bold]the loop")
    console.print(
        f"fleet success rate {report.initial_success_rate:.0%} → "
        f"[bold]{report.final_success_rate:.0%}[/] over {len(report.rounds)} rounds, "
        f"{report.versions_published} versions published, {report.rollbacks} rolled back"
    )
    console.print(f"dashboard written to {path}")


def _reasons(triage: dict) -> str:
    reasons = triage.get("reasons", {})
    if not reasons:
        return ""
    return "[dim]" + " ".join(f"{k}:{v}" for k, v in sorted(reasons.items())) + "[/]"


@app.command()
def regression(
    root: str = RootOpt,
    rounds: int = typer.Option(4),
    episodes: int = typer.Option(60),
) -> None:
    """Ship a release that every pre-flight check approves, and is worse.

    The incident is a plausible one: a pipeline change drops the fleet shards,
    and the release is trained on the factory dataset alone. It loads, it runs
    inside the latency budget, it passes the device health gate, and it scores
    *better than the incumbent* in the hub's simulator — because the hub's
    simulator is the nominal robot, and on the nominal robot it genuinely is
    fine. Nothing at the hub can see the problem. Only the nodes can.
    """
    shutil.rmtree(root, ignore_errors=True)
    config = FleetConfig(root=Path(root), rounds=rounds, episodes_per_round=episodes)
    report, hub, nodes = run_loop(config)

    console.rule("[bold]a release that passes every check and is worse")
    incumbent = hub.policy_of(hub.current_version)
    regressed, _ = bootstrap_policy(config.environments[0], seed=99)

    shippable, rates, why = hub.worth_shipping(regressed)
    console.print(f"[dim]│[/] hub pre-flight: {why}")
    console.print(
        f"[dim]│[/] verdict [{'green' if shippable else 'red'}]"
        f"{'ships' if shippable else 'withheld'}[/] — "
        "the hub has a model of the nominal robot and of nothing else"
    )
    console.print(
        "[dim]│[/] [dim]ground truth the hub cannot see: "
        + ", ".join(f"{e.name} {success_rate(regressed, e, 40):.0%}" for e in config.environments)
        + " (incumbent: "
        + ", ".join(f"{success_rate(incumbent, e, 40):.0%}" for e in config.environments)
        + ")[/]"
    )

    next_version = max(hub._known_versions(), default=0) + 1
    percent = hub.canary_percent_for([n.config.node_id for n in nodes], next_version)
    published = hub.publish(
        regressed,
        rollout_percent=percent,
        notes="retrained without the fleet shards (simulated pipeline regression)",
    )
    console.print(f"[dim]│[/] published v{published} to {percent}% of the fleet")

    for _ in range(2):
        for node in nodes:
            node.poll_updates()
            node.run_episodes(episodes)
            hub.report_outcomes(node.take_outcomes())

    canary = hub.evaluate_canary()
    console.print(f"[dim]│[/] {canary.summary()}")
    for node_id, stats in sorted(canary.per_node.items()):
        console.print(
            f"[dim]│[/]   {node_id:24s} "
            f"v{canary.incumbent_version} {stats['incumbent']:.0%} "
            f"({stats['n_incumbent']}) → v{canary.candidate_version} "
            f"{stats['candidate']:.0%} ({stats['n_candidate']})"
        )

    if canary.verdict == "ROLLBACK":
        restored = hub.current_version
        version = hub.roll_back(canary)
        console.print(
            f"[dim]│[/] [green]rolled back[/] — published v{version}, which carries "
            f"v{restored}'s policy. Versions only ever go up; undoing a release "
            "means publishing the old policy under a new number."
        )
        for node in nodes:
            node.poll_updates()
        console.print(
            "[dim]│[/] nodes now on: "
            + ", ".join(f"{n.config.node_id}=v{n.policy_version}" for n in nodes)
        )
    else:
        console.print(f"[red]the gate did not catch it: {canary.verdict}[/]")
        raise typer.Exit(code=1)


@app.command()
def show(root: str = RootOpt) -> None:
    """Print the release log from a previous run."""
    path = Path(root) / "hub" / "releases.json"
    if not path.exists():
        console.print(f"[red]no release log at {path}[/] — run `fleet-loop run` first")
        raise typer.Exit(code=1)

    table = Table(title="releases")
    table.add_column("v", justify="right")
    table.add_column("rollout", justify="right")
    table.add_column("state")
    table.add_column("dataset")
    table.add_column("sim")
    table.add_column("notes", overflow="fold")
    for record in json.loads(path.read_text(encoding="utf-8")):
        table.add_row(
            str(record["version"]),
            f"{record['rollout_percent']}%",
            record["state"],
            (record.get("dataset_hash") or "—")[:12],
            f"{record['sim_success_rate']:.0%}" if record.get("sim_success_rate") else "—",
            record.get("notes", ""),
        )
    console.print(table)
