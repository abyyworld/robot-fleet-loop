# robot-fleet-loop

**A robot fleet that improves itself from its own data, with a gate that can
tell the difference and undo the release when it cannot.**

Three nodes in different conditions run a policy, decide locally which of their
episodes are worth a person's attention, ship that under a hard bandwidth
budget. The hub validates, retrains, canaries the result to part of the fleet,
and promotes or rolls back on what the fleet actually reports.

```bash
git clone https://github.com/abyyworld/robot-fleet-loop
cd robot-fleet-loop
make install
make anchors     # confirm the task discriminates before believing any number
make loop        # the whole loop, eight rounds
make rollback    # ship a release every pre-flight check approves, and watch it come back
make test        # 94 tests
```

Built on [`edge-policy-runtime`](https://github.com/abyyworld/edge-policy-runtime) —
the signed over-the-air down-link, the device health gate and the telemetry
up-link are that repo's, and nothing here reimplements them.

---

## The result

```
round 1  fleet 32%
│ node-01-nominal          v1    87%  sent 24/ 60 ( 39 KiB, 100% of budget)  failed:5 novel:20
│ node-02-miscalibrated    v1     8%  sent 10/ 60 ( 33 KiB,  84% of budget)  failed:5 novel:5
│ node-03-long-reach       v1     0%  sent  6/ 60 ( 34 KiB,  87% of budget)  failed:6
│ no retrain — 40 new episodes since the last training set; need 100
...
round 4  fleet 56%
│ promoted v2 to the whole fleet — v2 vs v1: +38.6% [+31.1%, +46.4%] → PROMOTE
...
round 7  fleet 96%
│ promoted v3 to the whole fleet — v3 vs v2: +50.6% [+44.2%, +56.4%] → PROMOTE

fleet success rate 32% → 96% over 8 rounds, 3 versions published, 0 rolled back
hub: 416 episodes from 58 shards (852 KiB), 1440 outcomes, 3 dataset versions
```

Round 1 is the premise. The factory policy was trained before deployment on the
nominal robot's conditions, and it is genuinely good there — 87%. On the node
with a worn joint it scores 8%, and on the node working outside the training
workspace it scores 0%. Nothing is broken; the arms are fine and the task is
solvable, as the expert's 100% everywhere shows. The deployed policy simply
never saw those states, so it extrapolates, and a policy extrapolating is a
policy guessing. **No amount of examining v1's training metrics would have
revealed it. Only running it there does.**

Eight rounds later the fleet is at 96%, on **16% of the states it observed**,
with every release traceable to the dataset that produced it and every promotion
made on a confidence interval rather than a timer.

## What is actually hard here

### 1. Choosing what to upload

The constraint people name first is bandwidth. The one that binds is
**labelling**: the states a policy struggles in are the states somebody has to
teleoperate a correction for, and a person's time costs more than a cellular
modem's. So triage is not "compress the log" — it is "choose which states are
worth a person's attention", and the ranking is the product.

| signal | what it catches |
| --- | --- |
| `failed` | states the current policy provably cannot handle |
| `near_miss` | where the policy is about to fail — invisible in a success rate until it crosses over |
| `device_flagged` | the on-device detectors fired: out-of-distribution observation, saturated action, inference failure |
| `novel` | a region the accepted dataset has little of; two hundred more of the same failure teach nothing |

**And a reserved share of ordinary successes, deliberately.** If the fleet only
ever uploads failures, the training set drifts to contain nothing but hard
states, the retrained policy is optimised for a distribution the robot is not
in, and each round makes the skew worse — a feedback loop where the data
selection rule poisons the model that generates the next round of data. A fixed
fraction of the budget is reserved for uniformly-sampled ordinary episodes,
filled *before* anything is ranked and *not* from the top of the ranking (the
highest-scoring successes are the near-misses, which are not ordinary and do not
correct the skew).

#### The bug: a fleet succeeding 47% of the time, uploading 88% successes

The first version scored value per *episode* and ranked by value per byte. That
looks obviously correct and is a length penalty in disguise: an episode's size
is proportional to its steps, a failure runs to the step limit and a success
settles in a tenth of that, so a 120-step failure had to be ten times more
valuable than a 12-step success merely to break even. It never was. The measured
result was a fleet succeeding 47% of the time uploading a training set that was
**88% successes** — the exact inversion of what triage is for.

The fix is not a tuned weight. The thing being bought with bandwidth is *states
worth labelling*, and a failure that took 120 steps to fail contains 120 of them,
so the weights are per state and an episode's value is the weight times its
length. Value per byte then compares information density rather than episode
length, which is what the knapsack wanted in the first place.

### 2. Two channels, and never confusing them

| | outcomes | trajectories |
| --- | --- | --- |
| size | ~80 bytes | tens of kilobytes |
| sent | **always, every episode** | only what triage selects |
| used for | the canary gate | retraining |

A gate whose input is filtered by a value heuristic is measuring the heuristic.
This is the single most useful decision in the repo: 1440 outcomes reached the
hub against 416 trajectories, so the comparison that decides promotion is
computed on *everything the fleet did*, while the bytes are spent only on what is
worth labelling.

### 3. Selective sync under a real budget

The budget is in **wire bytes**, because that is what a data plan is priced in.
Triage ranks by in-memory float32 size, which is what it can see. The node keeps
a running estimate of the gap, budgets triage against it, enforces the true wire
budget after packing, and reports the estimate's error rather than hiding it.

Two measurements changed the encoding:

- **Deflate on float32 trajectory data is roughly 1.0x**, sometimes below 1 once
  archive framing is counted. Real-valued sensor data has high-entropy low-order
  mantissa bits and there is nothing for a dictionary coder to find. The "these
  compress about 3x" assumption this was built on was simply wrong.
- **One array pair per episode meant a zip member header per episode**, and at
  eight episodes of ten steps the framing was a fifth of the shard.

What ships is float16 in two concatenated arrays with a length index: 1.6–2.2x
smaller than in-memory float32, almost entirely from the dtype. The precision
cost is bounded and tested — worst-case round-trip error is 1.6e-3 rad on a
joint angle whose success tolerance is 0.10 rad, and 6e-5 rad on an action whose
rate cap is 0.15. Encoders do not resolve better than that, and shipping bits
below the sensor's noise floor is paying to transmit noise.

Manifests are offered before payloads, so a node retrying after a lost response
pays a few hundred bytes rather than tens of kilobytes. Shards are
content-addressed, and a shard is dropped from the node only once the hub
acknowledges it — at-least-once, same contract as the telemetry up-link.

### 4. Not believing the fleet

Data arriving from a fleet is the least trustworthy input in the system: it
crossed a link, it was produced by software the hub may not have deployed, on
hardware with a clock nobody set, and it is used to train the policy that goes
back down to every node. One corrupted shard in the training set is a
fleet-wide regression with no obvious cause.

Checked before anything is accepted: schema version, observation and action
contract by name, shapes, finiteness, actions within what the actuator can carry
out, clock skew, and **whether the policy version is one this hub published** —
an episode that cannot be attributed to a release cannot judge one or explain one
later.

**Rejected shards are quarantined, not dropped.** A drop is invisible. The most
valuable thing a validation layer produces is not clean data; it is the list of
what was dirty and why, because that list is how you find the node with the
failing sensor.

### 5. Relabelling, and why the node uploads actions at all

- A **failed** episode's actions did not work. Training on them teaches the
  policy to reproduce the failure. What is valuable is the *states* it visited,
  and what they need is a correct action attached to each — the DAgger step. In a
  real fleet those corrections come from a teleoperator; here an expert
  controller stands in, which makes this step considerably easier than reality.
- A **successful** episode's actions did work, in that node's conditions,
  including whatever miscalibration that node has. They are demonstrations and
  are used as recorded.

That is why actions are uploaded and not only observations: without them there
is nothing to keep from the successes, and no way to measure how far the policy
was from the correction where it failed. `mean_correction` is that measurement —
how wrong the deployed policy is, in radians, in the states it is actually
failing in — and it is the most useful single number the hub produces.

Datasets are content-addressed by their shard ids plus the labelling rule, so
every checkpoint names the data that produced it. The set is windowed, and the
policy-version mix is recorded, so drift is visible rather than assumed away.

### 6. Retraining on a reason

Both conditions must hold: enough new episodes, **and** enough of them from
shards the last training set did not have. Retraining on data the last run
already saw costs a training run, an evaluation, a canary cohort and a promotion
decision — a full cycle of fleet exposure — to learn nothing.

### 7. The decision the device cannot make

`edge-policy-runtime`'s health gate catches broken, incompatible and too slow. It
cannot catch *runs fine, succeeds less often*: that needs episodes and a
comparison against the incumbent, neither of which exists at activation time.

**`make rollback` is the case that matters.** The incident is plausible — a
pipeline change drops the fleet shards and a release is trained on the factory
dataset alone:

```
│ hub pre-flight: 100% vs the incumbent's 100% in simulation
│ verdict ships — the hub has a model of the nominal robot and of nothing else
│ ground truth the hub cannot see: nominal 100%, miscalibrated 52%, long-reach 0%
│                       (incumbent:  100%,                65%,             20%)
│ published v3 to 37% of the fleet
│ v3 vs v2: -14.6% [-23.7%, -5.4%] → ROLLBACK
│   node-02-miscalibrated    v2 65% (60) → v3 48% (120)
│   node-03-long-reach       v2 17% (60) → v3  4% (120)
│ rolled back — published v4, which carries v2's policy
│ nodes now on: node-01=v4, node-02=v4, node-03=v4
```

It loads. It runs inside the latency budget. It passes the device health gate on
every node. It scores as well as the incumbent in the hub's simulator — because
the hub's simulator is the *nominal* robot, and on the nominal robot it genuinely
is fine. What each node's arm has actually become is precisely what nobody wrote
down. Only the fleet can tell.

#### How the comparison is made, and what it cannot do

**Within-node, before versus after.** Comparing canary nodes against non-canary
nodes is an observational study across different hardware in different
conditions, and with a handful of nodes the hardware difference dwarfs the
policy difference. Restricting to nodes that ran both versions removes node
identity as a confounder entirely.

**The residual confound is time, and it is not removed.** A before/after
comparison attributes to the release anything else that changed between the two
windows. The mitigation is that canary and non-canary nodes are observed over the
same wall-clock window, so a fleet-wide temporal effect appears in both arms —
a mitigation, not a fix.

**Nodes are weighted equally, not by episode count**, so a busy node cannot
decide the fleet's verdict — and on a heterogeneous fleet that weighting asks
"does this work everywhere" rather than "does this work on average". A candidate
that holds its overall rate while collapsing on one node is refused; that is the
failure aggregates hide, and on a heterogeneous fleet it is the likely shape of a
bad release.

**Three verdicts.** `INCONCLUSIVE` is the one most rollout systems lack. A
comparison whose interval is wider than the tolerance it is testing has not shown
the release is safe; it has shown the study was too small, and the report says
how many episodes per arm would be needed instead.

**The statistics are weaker than an offline comparison, unavoidably.** An
offline harness evaluates two policies on matched seeds and pairs the outcomes,
which removes episode difficulty from the comparison entirely. Matched seeds do
not exist in a fleet — two nodes never face the same episode — so this is
unpaired, and `make power` shows the price: resolving a 3-point difference around
a 70% baseline needs over 3,000 episodes per arm.

### 8. A three-node fleet cannot run a 10% canary

The smallest cohort that supports a within-node comparison on two nodes is
two-thirds of a three-node fleet. So the hub asks what rollout percentage
actually covers enough nodes rather than picking a number that sounds cautious
and produces `INCONCLUSIVE` forever. The tension between limiting exposure and
having the power to detect a regression is real and does not go away with fleet
size — it only gets cheaper. On five hundred nodes the same two-node minimum is
0.4%.

### 9. Rolling back means publishing forward

Versions are monotonic. The device's only question is "is this newer than what I
run", and a scheme where the answer can go backwards is a scheme where a device
that missed a poll ends up on the wrong side of a rollback forever. Undoing v3
publishes v4 carrying v2's policy, with the reason in the notes.

### 10. The dashboard shows skew, not intentions

Version skew is the normal state of a fleet, not an anomaly: nodes are offline,
in a cohort that has not been reached, or sitting on a release they refused. A
dashboard that shows the published version and calls it the fleet's version is
showing you the hub's intentions.

```
fleet — published v3, canary v4                        skew: v3, v4
node                   conditions      running  success  p99 ms  backlog  dropped  refused
node-01-nominal        nominal              v3     100%    0.01      221        —        —
node-02-miscalibrated  miscalibrated        v3     100%    0.01      388        —        —
node-03-long-reach     long-reach           v3      87%    0.01      355       61        —
```

`backlog` and `dropped` are the fleet's blind spot: a node that has quietly been
discarding its most interesting episodes for a week looks perfectly healthy
without them.

## Honest limits

- **The task is kinematic.** Joints integrate commanded deltas subject to limits
  and a rate cap. No contacts, no dynamics, no friction. Success rates are
  harness results, not robot results. What is being demonstrated is the loop, and
  every part of it is indifferent to what is on the other side of `reset` and
  `step`.
- **The teleoperator is a proportional controller.** In a real fleet, corrections
  for flagged states cost a human's time and are imperfect. Here they are free
  and exact, which makes the relabelling step easier than reality by a wide
  margin. The loop's *mechanics* are the claim; its sample efficiency is not.
- **Three nodes is not a fleet.** Every statistical statement here is limited by
  it, and §8 is where that shows most.
- **The hub and the nodes are one process.** The wire format is real and
  everything crosses it, but there is no second HTTP server here —
  `edge-policy-runtime` already demonstrates both links over real HTTP, and the
  `HubEndpoint` protocol is three methods wide precisely so it can go behind one.
- **The validation split leaks.** Consecutive steps within an episode are highly
  correlated and the split is by sample, so the validation loss is optimistic
  about generalising to *new* episodes. It is used for early stopping, not for
  promotion — that is what the canary is for — and saying so beats reporting a
  number that looks like generalisation and is not.

## Layout

```
src/fleet_loop/
  wire.py            the contract: outcomes, trajectories, shards, sync plans
  sim.py             the task, three node conditions, and the expert that relabels
  node/
    runner.py        runs episodes, logs, bounded value-ordered local store
    triage.py        what is worth a person's attention, and the success quota
    sync.py          bandwidth-aware selective upload, at-least-once
  hub/
    ingest.py        validate, deduplicate, quarantine with a reason
    dataset.py       content-addressed assembly, DAgger relabelling, windowing
    train.py         behaviour cloning in numpy, normalisation folded in
    canary.py        within-node comparison, three verdicts, power
    hub.py           retrain triggers, pre-flight, publish, promote, roll back
  dashboard.py       per-node health and version skew; terminal and static HTML
  loop.py            the scheduler
tests/               94 tests
```

## Commands

```bash
make anchors      # trivial policies must fail and the expert must succeed
make loop         # eight rounds, three nodes, end to end
make rollback     # a release that passes every check and is worse
make power        # episodes per arm a fleet comparison needs

fleet-loop run --rounds 12 --budget 20000    # tighter link, slower convergence
fleet-loop show                              # the release log from a previous run
```

## Status

Working end to end. The fleet improves from its own data, every release is
traceable to a dataset, promotion happens on an interval rather than a timer,
and a regression that no pre-flight check can see is caught by the fleet and
undone — all asserted in tests rather than described. The simulator and the
teleoperator stand-in are placeholders and say so; the triage, the sync
protocol, the validation layer, the canary gate and the release machinery are
not.

## Licence

MIT.
