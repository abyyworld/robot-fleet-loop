"""What an operator looks at.

The three questions a fleet dashboard has to answer, in order:

1. **What version is each node actually on?** Not what was published — what is
   running. Version skew is the normal state of a fleet, not an anomaly: nodes
   are offline, in a rollout cohort that has not been reached, or sitting on a
   release they refused. A dashboard that shows the published version and calls
   it the fleet's version is showing you the hub's intentions.
2. **Which nodes are unhealthy, and is the release why?** Health and version have
   to be on the same row. Split across two views they are two facts; together
   they are a hypothesis.
3. **What is not being told to anyone?** Pending upload backlog and locally
   dropped episodes are the fleet's blind spot, and they are invisible unless
   something puts them on the screen. A node that has quietly been discarding
   its most interesting episodes for a week looks perfectly healthy.

Rendered as a Rich table for a terminal and as a self-contained HTML file for
anywhere else. No JavaScript, no server: an operations page that needs a build
step is an operations page that is out of date.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .hub import Hub
from .node import FleetNode


def fleet_state(hub: Hub, nodes: list[FleetNode]) -> dict:
    snapshots = [node.snapshot() for node in nodes]
    versions = {s["policy_version"] for s in snapshots}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hub": hub.status(),
        "nodes": snapshots,
        "version_skew": len(versions) > 1,
        "versions_running": sorted(versions),
        "published_version": hub.current_version,
        "canary_version": hub.candidate_version,
        "releases": hub.release_log,
    }


def render(hub: Hub, nodes: list[FleetNode], console: Console | None = None) -> dict:
    """Print the dashboard. Returns the state it rendered."""
    console = console or Console()
    state = fleet_state(hub, nodes)

    header = f"published v{state['published_version']}"
    if state["canary_version"]:
        header += f", canary v{state['canary_version']}"
    if state["version_skew"]:
        header += f"  [yellow]skew: {', '.join(f'v{v}' for v in state['versions_running'])}[/]"

    table = Table(title=f"fleet — {header}")
    table.add_column("node")
    table.add_column("conditions")
    table.add_column("running", justify="right")
    table.add_column("success", justify="right")
    table.add_column("p99 ms", justify="right")
    table.add_column("backlog", justify="right")
    table.add_column("dropped", justify="right")
    table.add_column("refused")

    for snapshot in state["nodes"]:
        rate = snapshot["recent_success_rate"]
        rate_cell = "—" if rate is None else f"{rate:.0%}"
        if rate is not None and rate < 0.5:
            rate_cell = f"[red]{rate_cell}[/]"

        version_cell = f"v{snapshot['policy_version']}"
        if snapshot["policy_version"] != state["published_version"]:
            version_cell = f"[yellow]{version_cell}[/]"

        refused = ", ".join(f"v{v}" for v in snapshot["quarantined"]) or "—"
        if snapshot["updates_frozen"]:
            refused = f"[red]frozen[/] {refused}"

        table.add_row(
            snapshot["node_id"],
            snapshot["environment"],
            version_cell,
            rate_cell,
            f"{snapshot['latency_p99_ms']:.2f}" if snapshot["latency_p99_ms"] else "—",
            f"{snapshot['pending_episodes']}",
            f"{snapshot['dropped_locally']}" if snapshot["dropped_locally"] else "—",
            refused,
        )
    console.print(table)

    ingest = state["hub"]["ingest"]
    console.print(
        f"hub: {ingest['episodes']} episodes from {ingest['shards']} shards "
        f"({ingest['bytes_received'] / 1024:.0f} KiB), "
        f"{ingest['outcomes']} outcomes, "
        f"{state['hub']['datasets']} dataset versions"
    )
    if ingest["quarantined"]:
        console.print(
            f"[yellow]quarantined:[/] {ingest['quarantined']} shards — "
            + ", ".join(f"{k}={v}" for k, v in sorted(ingest["quarantine_by_reason"].items()))
        )
    return state


# -- static export -----------------------------------------------------------


def write_html(state: dict, path: Path | str) -> Path:
    """A single self-contained file. No server, no build, no stale bundle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for snapshot in state["nodes"]:
        rate = snapshot["recent_success_rate"]
        skewed = snapshot["policy_version"] != state["published_version"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(snapshot['node_id'])}</td>"
            f"<td>{html.escape(snapshot['environment'])}</td>"
            f"<td class='{'warn' if skewed else ''}'>v{snapshot['policy_version']}</td>"
            f"<td class='{'bad' if rate is not None and rate < 0.5 else ''}'>"
            f"{'—' if rate is None else f'{rate:.0%}'}</td>"
            f"<td>{snapshot['latency_p99_ms'] or '—'}</td>"
            f"<td>{snapshot['pending_episodes']}</td>"
            f"<td>{snapshot['dropped_locally'] or '—'}</td>"
            f"<td>{', '.join(f'v{v}' for v in snapshot['quarantined']) or '—'}</td>"
            "</tr>"
        )

    releases = "".join(
        "<tr>"
        f"<td>v{r['version']}</td>"
        f"<td>{r['rollout_percent']}%</td>"
        f"<td>{html.escape(str(r['state']))}</td>"
        f"<td>{html.escape(str(r.get('dataset_hash') or '—'))[:12]}</td>"
        f"<td>{html.escape(str(r.get('notes', '')))}</td>"
        "</tr>"
        for r in state["releases"]
    )

    path.write_text(
        f"""<!doctype html>
<meta charset="utf-8">
<title>fleet</title>
<style>
 body {{ font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; margin: 2rem; }}
 h1 {{ font-size: 1.1rem; }}
 table {{ border-collapse: collapse; margin-bottom: 2rem; }}
 th, td {{ border-bottom: 1px solid #ddd; padding: .35rem .8rem; text-align: left; }}
 th {{ font-weight: 600; }}
 .warn {{ color: #a60; }}
 .bad {{ color: #c00; font-weight: 600; }}
 .meta {{ color: #666; }}
 @media (prefers-color-scheme: dark) {{
   body {{ background: #111; color: #ddd; }}
   th, td {{ border-color: #333; }}
   .meta {{ color: #999; }}
   .warn {{ color: #d90; }}
   .bad {{ color: #f66; }}
 }}
</style>
<h1>fleet — published v{state["published_version"]}
{f", canary v{state['canary_version']}" if state["canary_version"] else ""}</h1>
<p class="meta">{html.escape(state["generated_at"])} ·
{"version skew: " + ", ".join(f"v{v}" for v in state["versions_running"]) if state["version_skew"] else "all nodes on the published version"}</p>
<table>
<tr><th>node<th>conditions<th>running<th>success<th>p99 ms<th>backlog<th>dropped<th>refused</tr>
{"".join(rows)}
</table>
<h1>releases</h1>
<table>
<tr><th>version<th>rollout<th>state<th>dataset<th>notes</tr>
{releases}
</table>
<h1>hub</h1>
<pre>{html.escape(json.dumps(state["hub"], indent=2))}</pre>
""",
        encoding="utf-8",
    )
    return path
