# Why the hub and the nodes share a library

`robot-fleet-loop` depends on `edge-policy-runtime` and uses it on both sides: the
nodes run its OTA client, its inference runtime and its edge-case detectors, and
the hub builds releases with its bundle format and signing. Nothing is
duplicated. That is a deliberate choice and it is the wrong one at a certain
scale, so this is where the boundary is.

## What the choice buys

**One definition of the release format.** A bundle's manifest names the
observation contract, the health envelope, the file hashes and the provenance. If
the hub and the device each had their own copy of that schema, the failure mode
is not a crash — it is a device that verifies a bundle successfully and then
interprets one field differently from the party that signed it. That class of bug
is silent, it survives testing, and it appears in the field on the one release
where the two definitions drifted.

**The gate the device runs is the gate the hub built for.** `HealthSpec` travels
inside the bundle. The hub sets the latency budget and the action-deviation
tolerance; the device measures itself against them. With two schemas that
agreement is a convention; with one type it is a compile-time fact.

**The device half is genuinely done.** The staging, verification, atomic
activation, quarantine and rollback machinery in `edge-policy-runtime` is about
five hundred lines with ninety tests behind it. Re-implementing a lesser version
here to avoid a dependency would be worse code and a worse claim.

## What the choice costs, and when it stops being right

A shared library means the hub and every node must be able to run the *same
version of Python and the same package*. That is fine at three nodes and fine at
five hundred identical ones. It stops being fine when:

- **Devices and hub upgrade on different schedules.** They already do — that is
  what `schema_version` and `runtime_min_version` in the manifest are for — but a
  shared library makes it tempting to change a shared type and deploy both halves
  together, which is exactly the habit that produces an un-rollbackable release.
- **Nodes stop being Python.** A C++ or Rust node cannot import this. At that
  point the contract has to exist independently of any implementation of it.
- **The fleet is heterogeneous in hardware.** A shared package implies a shared
  build, and a Jetson, an x86 industrial PC and an ARM SBC do not share one.

## What it would become

The wire format — `Manifest`, `HealthSpec`, `EpisodeOutcome`, `ShardManifest`,
`SyncPlan` — moves into a schema package that generates types rather than
defining them: protobuf, or JSON Schema with codegen. The hub and each node
implementation depend on the *schema*, not on each other. Version negotiation
stops being a convention and becomes the thing the codegen enforces.

Two things do not change, and they are the reason the current arrangement is
survivable in the meantime:

1. **The manifest already carries `schema_version` and `runtime_min_version`,
   and both are checked before a bundle is accepted.** A device refuses a bundle
   it does not understand rather than guessing at it.
2. **`fleet_loop.wire` is a single module with no dependency on anything else in
   this repo, and `HubEndpoint` is three methods wide.** The seam is already
   drawn where it would need to be cut.

## What is *not* shared, deliberately

The hub does not import the node's triage, and the node does not import the
hub's ingest or canary logic. They exchange the types in `fleet_loop.wire` and
nothing else. The reason is the same one that makes the shared bundle format
safe: what crosses the boundary is data with a version on it, and each side is
free to change how it decides things without the other side needing to agree.

The one place this shows is that the hub's validator re-checks, against the
payload, everything the node's manifest claims — shapes, finiteness, action
bounds. The node could be trusted to have got that right, since it is running
code from the same package. It is not trusted, because in a year it will not be.
