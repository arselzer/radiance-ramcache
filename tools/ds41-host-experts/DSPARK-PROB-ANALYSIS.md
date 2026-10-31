# Offline DSpark probability analysis

`dspark_prob_analyze.py` reads step2A's server-log records. It loads no model and runs no engine,
network, simulator or GPU work. The preparation agent has not executed it or its tests. Run the
analysis/tests through the box-wide build lock and nice19; analysis can consume appreciable CPU
with the threshold grid and request-group bootstrap. No engine policy is implemented here.

## Inputs and mapping

First validate each capture, using unique run names to namespace sequence/epoch IDs:

```sh
python3 tools/ds41-host-experts/dspark_prob_analyze.py analyze \
  --trace code2=/path/code-Ds2-Dpc.server.log \
  --trace chat2=/path/chat-Ds2-Dpc.server.log --out /tmp/ds41-prob-validation.json
```

Validation requires one proposal and one terminal outcome for each `(run,seq,epoch,depth)`,
unchanged IDs/position/signals across that pair, and depth2's predecessor equal to depth1's ID.
It checks reached/accepted prefix consistency and refuses engine-reported proposal mismatches.
Missing outcomes fail by default; `--allow-incomplete` permits inspection but disables candidate
selection. Cancelled, invalid-signal, capped and other discarded observations are counted
separately. An unreached depth2 is never entered as a negative conditional calibration trial.

Legacy `ab_requests.py` result rows contain prompt names/families but omit scheduler sequence IDs.
Do not map by the order of the sequences that happened to draft: some requests may draft none.
Supply an explicit manifest:

```json
{
  "schema": "ds41-proposal-metadata-v1",
  "requests": [
    {"run": "code2", "seq": 1, "family": "code", "prompt_key": "code-01", "split_group": "code-01"},
    {"run": "chat2", "seq": 1, "family": "chat", "prompt_key": "chat-01", "split_group": "chat-01"}
  ]
}
```

For a verified single-stream harness with no hidden generation/warmup requests, a helper creates
the manifest from ALL request rows in order, checks `concurrency==1`, failures and available stats
IDs, and records the caller's explicit assertion:

```sh
python3 tools/ds41-host-experts/dspark_prob_analyze.py make-metadata \
  --results code2=/path/code-Ds2-Dpc.json --results chat2=/path/chat-Ds2-Dpc.json \
  --assert-sequential --first-seq 1 --out /tmp/ds41-prob-metadata.json
```

That assertion cannot be proven from result JSON alone. Use manual IDs if uncertain. Review the
manifest before fitting. Repeated versions/runs of a prompt must share `split_group`; related
conversation turns should share a session group. It defaults to `prompt_key`. A deterministic
SHA256 split with a recorded seed holds out25% of groups, never individual tokens/epochs. Explicit
`"split":"train"`/`"heldout"` is accepted, with contradictory assignments to a group refused.
Freeze groups before reading outcome-dependent results. `family:"all"` is reserved internally.

## Calibration and measured costs

Pass `--metadata` to get equal-width `candidate_p` bins by family, depth and train/held-out split.
Only valid, non-cancelled, reached outcomes enter these bins. The binomial Wilson interval is
descriptive: it assumes independent trials and does not account for within-request correlation.
The held-out policy comparison instead bootstraps WHOLE split groups. A separate dispositions
table by family/depth/split retains capped, unreached, cancelled, invalid and missing outcomes.

Threshold fitting additionally requires a measured cost file. This template is intentionally
unusable until the nulls and source are replaced by actual matched measurements:

```json
{
  "schema": "ds41-proposal-costs-v1",
  "measurement_profile": "engine commit,256GB/DDR4-2666,configured cache budget and exact controls",
  "same_drafter_observer_wait": true,
  "families": {
    "code": {"c0_ms": null, "c1_ms": null, "c2_ms": null, "source": "window/arms and counter definition"},
    "chat": {"c0_ms": null, "c1_ms": null, "c2_ms": null, "source": "window/arms and counter definition"}
  }
}
```

Each Ck is the complete cost of one decision: query the SAME drafter and observer, take the SAME
early readback wait, then verify k proposals. **C0 still queries and observes each step. Plain
decode and existing Ds0/compute-OFF are not this C0.** Include the observer and wait's overlap loss;
measure verbose capture logging overhead separately. Do not silently mix128GB/pre-sync historical
costs with current256GB data, or combine incompatible cache/configuration profiles. The profile,
sources and matching assertion are operator-supplied provenance, not automatically verified facts.
No RAM/cache capacity is hardcoded in the tool. A deliberate `families.all` entry can supply one
shared measured profile; otherwise every observed family needs its own costs.

```sh
python3 tools/ds41-host-experts/dspark_prob_analyze.py analyze \
  --trace code2=/path/code-Ds2-Dpc.server.log --trace chat2=/path/chat-Ds2-Dpc.server.log \
  --metadata /tmp/ds41-prob-metadata.json --costs /tmp/ds41-prob-costs.json \
  --out /tmp/ds41-prob-analysis.json
```

The policy form being fitted is `p1<t1 -> verify0`, otherwise `p2<t2 -> verify1`, otherwise
`verify2`. Here p1/p2 are the candidate-set signal, not acceptance probabilities. The grid uses
training-signal quantiles plus0 (always eligible) and2 (never take that depth), retaining resolution
when shares cluster near1.0. Held-out signal values do not choose cut points. **Historical0.49/0.70 acceptance break-evens are not candidate_p
threshold defaults.** All threshold choices and the fixed0/1/2 reference choice use TRAIN groups
only. The selected thresholds are frozen and evaluated once on held-out groups. Defaults require
four training groups, two held-out groups and40 epochs in each split; small data reports insufficient
support. Family-specific fits are diagnostics: the current engine has no code/chat semantic
classifier, and the harness sends both families as completions. The `all` fit is one global rule.

The replay uses only epochs whose first TWO proposals have matching verified outcomes. A second
proposal after a rejected first is untested conditionally, but the accepted prefix length is still
known: the action2 local yield is1, not a fabricated second rejection. Per-action local yields are
`1`, `1+a1`, `1+a1+a2`. The objective is sum(local yields)/sum(measured Ck), not a mean of rates.
By default it drops the first/last eligible epoch per request to avoid prefill and final-token
budget edges; exclusions and conditional depth2 sample counts are reported. Calibration retains
valid boundary trials because it describes verification outcomes, not emitted-token yield.

This is **one-decision counterfactual replay**, not full-trajectory simulation: choosing a smaller
depth changes later query anchors and the proposals that will exist. A positive held-out replay
result selects a candidate for a real step2B window; it is not an achieved end-to-end speedup. The
report lists all held-out fixed-depth results but does not choose a new reference after seeing them.
Do not retune on this held-out set and continue calling it held out; reserve new prompts for a
second selection round.

Prepared host tests: `dspark_prob_analyze_test.py` (pair integrity, unreached/cancelled/discarded
separation, missing/mismatched data, group split invariance, train-only selection, boundary trim,
metadata leakage and unmatched costs). They have not been run by the preparation agent.

## Smallest step2B decision and lifetime plan

Source-only matched fixed controls are now described in
[DSPARK-EARLY-CONTROLS.md](DSPARK-EARLY-CONTROLS.md). They preserve the Db2 query/hot set and use the
existing `step_end_` event; they implement no probability decision. Their quiet C0/C1/C2 counters
must be measured before fitting a cost policy. The plan below applies to a later policy.

The current read-only capture is retired in `finish_block_drafter`, AFTER `issue_ahead` has planned
and issued the next target verification. The existing CONF score is from a separate prediction
head. Neither is a safe source for changing the depth of the proposal already issued.

After capture data exists and a candidate is chosen, the smallest host policy can be a new
default-off branch:

1. Keep query generation, selector IDs and observer arithmetic unchanged. Queue observations AND
   the existing StepSlot snapshot on rank0's query stream. Reuse the existing `step_end_` after both
   copies; an observation without its matching IDs/ctx is insufficient.
2. In the policy branch only, wait for that event BEFORE `expect_draft`/`step_next` in
   `issue_ahead`. Bind each decision to the BlockRow sequence/card and this query's epoch/actual
   StepSlot ctx/ID prefix. Cancellation/reuse must discard it, not use a previous query's score.
   This wait loses overlap; matched C0/C1/C2 must include that loss. Preserve the off-path issue
   order and late readback exactly.
3. Store the chosen0/1/2 count separately from `block_steps_` (the unchanged generated length).
   Pass that SAME count to `expect_draft` before building the next batch and to `set_draft` when
   publishing the original proposal PREFIX. Its exact-count check must still pass. Do not mutate
   global maximum depth, short-block shape, slot ABI layout, target acceptance or cache numerics.
4. A verify0 action still computes the drafter/observer next step, so a fresh signal is available.
   Keep the current one-sequence, request eligibility, context/budget and cancellation gates.
   With the bank's required lookup0, an empty proposal must remain empty. Missing/invalid signals
   need an explicit conservative fallback, tested separately; do not borrow the prior epoch.
5. Add prefix-count and cancellation/issue-ahead host cases, then one default-off byte gate and
   matched fixed0/fixed1/fixed2 controls with the SAME early wait. Compare chosen global policy
   against the frozen reference on new held-out code/chat requests, including total timings and
   bank cost beside concurrency. No engine policy work starts from replay gains alone.
