# DSpark proposal observations (step 2A)

This is a default-off, read-only instrument. It changes no selected token, draft count, depth
decision, verification, acceptance rule or reply. It does not implement the cost policy.

Set `RADIANCE_EXPERIMENTAL_DS41_DSPARK_PROB_CAPTURE=1`, or add the window arm part `Dpc` beside
an existing drafting `Ds`/`Ds1`/`Ds2` part. `Dpc` without a drafting part, including `Ds0`, is
refused. Each other arm explicitly sets the switch to 0. Unset/0 declares no observation
buffer/op and issues no observation or readback. The architecture and core must both support
it; build/install every plugin from the same build. No plugin ABI struct layout changes.

## What the signal means

For position `l`, let `C_l` be the EXISTING global top-K candidate IDs from the drafter's base
head (16 for this checkpoint). The selector scores these candidates as

```
score_l(c) = unary_l(c) + dot(markov_w1[previous_id], markov_w2[c])
candidate_p = exp(score_l(selected) - max(score_l)) /
              sum(c in C_l, exp(score_l(c) - max(score_l)))
```

The first predecessor is the committed anchor; the second is the actual first selected token.
The generic observer also supports the selector's optional hidden projection. Scores use the
selector's exact lane products and reduction order; normalization uses F32 `expf` and a serial
candidate-order sum. Temperature is 1. `candidate_p` is the
selected candidate's normalized share WITHIN THIS SET, **not a full-vocabulary drafter
probability, target probability, or probability of acceptance**. Markov bias outside the set
has not been computed. The old expected-acceptance break-even values cannot be applied directly
to this signal: calibrate against the measured outcomes first.

`score` is the selected Markov-conditioned score. `margin` subtracts the largest other
candidate's score (0 for a winning tie; +infinity for a singleton). Nonfinite scores or an
inconsistent selected path or duplicate candidate IDs make the signals invalid; they never change
the path. The F32 ID
fields are exact for the supported vocabularies (at most 2^24). JSON encodes nonfinite values
as null and identifies a positive-infinite margin with `margin_unbounded`.

`dflash_select` and its chained kernel retain their original selection body. A separate
`dflash_observe` launch reads its output and recomputes only depths 1/2. Its F32 plane is
persistent, so extending the candidates' read lifetime and copying observations cannot expose
reused transient storage. Rank 0 copies that plane on the query stream before `step_end_`, next
to the existing StepSlot readback. Other ranks have no host readback. This extra kernel, copy
and logging can cost time; measure overhead with capture off/on before interpreting rates.

## Trace and pairing

The existing `<arm>.server.log` holds all records; there is no separate output path or model
data dump. Startup marker: `DSpark read-only proposal capture: on`. Each data line starts
`DSpark proposal capture: ` followed by a JSON object. The window requires the startup marker
for `Dpc` and forbids it for other arms.

Each `(seq, epoch, depth)` has a `proposal` event and one terminal `outcome` event. `epoch`
identifies an actual query pass, not a confidence-head estimate; depths are 1-based. `anchor_pos`
is the next verify's first input position, and `pos = anchor_pos + depth`. IDs and anchor come
from the same StepSlot as the proposal that `set_draft` publishes. Commit pairs position AND the
verified ID prefix, including a second proposal's predecessor. A mismatch is recorded rather
than borrowed as another epoch's result. Fixed per-request storage avoids per-step allocation.

Fields `candidate_count`, `candidate_p`, `score`, `margin`, `token`, `predecessor` and
`signal_valid` accompany both events. On a proposal event, outcome booleans are null.

- `verified`: a matching target verification result was observed at commit. This describes a
  result, not mere inclusion in a device batch. A cancelled in-flight request can still have one.
- `reached`: its predecessor proposals all accepted, so this conditional proposal was tested.
- `accepted`: true/false only for reached proposals; **null** for an unreached second proposal.
  At acceptance count 0, depth1 is rejected and depth2 unreached; at 1, depth1 accepted and
  depth2 rejected; at 2, both accepted.
- `cancelled`: cancellation suppressed normal emission; exclude these records when calibrating
  ordinary request performance. Cancellation before any verify gets an unverified terminal row.

Generated rows beyond the installed verify cap get `verify_cap`, never rejection. Not published,
superseded, held-without-verification, preempted, closed, reaped and shutdown proposals also get
terminal statuses. A sequence slot cannot transfer an observation to a new request. Shutdown
and final teardown flush pending records, so a normal run has one outcome for every proposal.
Captures contain no prompt or reply text.

## Review and validation queue

Short source review of `4f016272` and `88130277`: no blocker found. The bank's solo rule suppresses
both drafting and pending verification under multi-decode load; per-sequence tap history is
bounded by the drafter's carved cache slots; catch-up respects history/window floors; sequence-ID
checks and deferred release protect cancellation/reuse; the harness checks solo-mode startup
both ways. This was a focused source review, not a new runtime proof.

The preparation agent did not run builds, tests, emulators, benchmarks, servers or GPU work.
The integration owner must run the serialized queue under `work/AGENT-LOAD-RULES.md`:

1. Build `radiance`, `deepseek41`, `r4d`, `r4d_selftest` and the focused test targets from this
   worktree. New host test `draft_prob_test` is auto-discovered. The numerical oracle target is
   `r4d_emu_dprob` with a clang C++ build, or `r4d_emu_dprob_bin` for the custom-compiler path.
2. Through `flock work/build.lock nice -n19`, run ctest with `-LE gpu` and anchored names for
   `draft_prob_test`, `sched_test`, `place_test`, `ds41_dspark_defer_arch_test`, `r4d_emu_dprob`.
   Cases cover flags, ID/position/conditional-prefix matching, caps, cancellation, closed/reused
   slots, unchanged issue-ahead tokens/depths, persistent buffer/order, Dm placement and unknown
   readers. The SIMT oracle covers strides, missing hp, chained predecessors, ties, nonfinite
   inputs, singleton sets, bad shapes, input immutability and softmax/score/margin.
3. Inside ONE ordinary two-lock production stop/restore window, run the served libr4d's
   `dflash_observe_bf16` selftest (it runs unchanged selection too), then code and chat `Ds2`
   off/on pairs plus a merged-base reference. Optional `Ds1` pairs test the cap1 path too.
   Compare every generated reply and the existing verbose proposal-ID lines, require valid
   capture signals and exactly paired terminal records, and check depth/acceptance counters.
   Review capture overhead separately from policy benefit. This step adds no policy benefit.

Use the current 256 GB / 192 GiB-cache configuration for the primary window, with the existing
cache-budget flags. This implementation neither assumes that size nor changes startup budgeting,
miss paths or the useful 128 GB optimizations. Cache resizing and network tiers are outside scope.
Leave thresholds and cost-aware depth decisions for step 2B after captured code/chat data exists.
