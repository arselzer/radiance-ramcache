# Matched DSpark early-readback controls

These default-off controls measure fixed verified prefixes0/1/2. They implement no probability
policy, choose no thresholds, and claim no improvement. The preparation agent edited source only;
the new host/architecture cases, build and GPU gates still belong to the integration owner.

Set `RADIANCE_EXPERIMENTAL_DS41_DSPARK_EARLY_VERIFY=0`, `1` or `2`, or use the window parts
`Dme0`, `Dme1`, `Dme2`. Absent or `off` preserves the existing late-readback order. In particular,
**0 means a measured verify0 action that still queries, observes and waits every eligible step**;
existing `Ds0` / compute-OFF does not measure this action. Invalid values are refused.

Use bare `Ds` beside each Dme part: configured maximum depth, proposal pitch and native bank stay5.
Legacy `DsN` caps, `Dsa`, `Dc`, `Dl`, `Dg`, compute-OFF, trees, KLD/coupled verification, non-bf16 KV
and multi-sequence deployments are refused. The existing EP4 native-bank/final-placement authority
still applies, including static `DvN` hot sets. No RAM/cache budget is fixed by this instrument;
variable startup budgets and prior small-cache/miss paths remain available.

The full query stays5 rows, or an already configured fixed short query (`--draft-short-from1`,
2..4 rows) stays that shape. **Db2 remains a two-row query and two generated proposals for ALL
three controls.** It is not replaced by the full5 query, resized to the verification prefix, or
skipped at0. Use the SAME Dkt/Db2/Dv24/Df/Rb/Vp/Pmr and placement/cache settings as the capture
profile being priced. `Dx` can be retained consistently, but does not suppress queries at fixed0.

For example, take an otherwise identical existing single-stream code/chat arm, replace `Ds2` with
`Ds-Dme0`, `Ds-Dme1`, `Ds-Dme2`, and keep `Db2` plus every other part. The harness explicitly sets
the early flag to `off` in every other arm and checks the startup marker both ways, including k.
It sets aside a Dme result without positive steady/query/wait counters at shutdown. Both homes and
all plugins must belong to one rebuilt aql-sync/home-sync candidate, as for every DS41 window.

## Order, bindings and exclusions

The enabled order is:

1. Issue the current target step, unchanged block context/query, unchanged selector, optional
   observer, observation copy and whole StepSlot copy. `candidate_p` remains a temperature1
   softmax over the existing Markov-conditioned top-R candidates, NOT a full-vocabulary or
   acceptance probability. No fixed control reads it to choose k.
2. Record the EXISTING `step_end_`, then wait for it before `issue_ahead`. It covers both copied
   observations and the complete StepSlot. No new event or wait exists on the disabled path.
   Existing later collect/so.early waits stay in place; here they wait on already completed work.
3. Validate BlockRow request identity/card and actual StepSlot ctx against the current planned
   input plus actual acceptance; check observed IDs/predecessors against that SAME StepSlot.
   Reserve its epoch before issue-ahead. Preserve the entire generated tuple in fixed storage;
   compare it with the later publication readback again.
4. Use the SAME fixed k in `expect_draft`, `step_next` and `set_draft`; `block_steps_` remains the
   generated length. The original proposal prefix is published and the existing exact-count
   validation remains active. Normal planning also uses k when no step can issue ahead.
5. After commit/publication, require the same live request/card, actual committed ctx, output
   anchor token, installed count and original prefix IDs. Cancellation/reuse cannot validate a
   different query. Capture events use the epoch already reserved at the early boundary.

`B_t` is the host timestamp immediately after the early `step_end_` wait returns. A steady sample
`B_t..B_(t+1)` is tagged with the EARLIER query's `(seq,epoch,k)`: it includes that prefix's target
verification, the next block context/query/observer/copies/wait, and intervening host work.
`wait_ns` for the sample is the wait ending at `B_(t+1)`, already INCLUDED in `cycle_ns`, not an
additional cost to add. `cycle_ns / n` is the mean Ck, not tokens/s; accepted/emitted counters allow
a separate yield audit. These are host wall costs, not isolated GPU kernel times.

Only consecutive same-sequence/card, same-query-shape pure decode boundaries count. The current
verification must consume the earlier epoch's exact prefix at its actual ctx. Both requests must
remain live and publication must succeed; output must grow by exactly `1+accepted`. Prefill and
its first decode transition, missing queries, context clamps/prefix mismatches, end/EOS/budget
truncation, cancellation and unpublished cycles are excluded. An unclosed tail epoch is reported
separately. Queries, observer copies and waits count even when the corresponding cycle is excluded.
End effects still belong in end-to-end arm timings; steady Ck alone does not forecast short replies.

Invalid NUMERICAL observer signals do not change fixed k, query execution, publication or replies.
The observer echoes IDs for every present depth before numerical validation; NaN/nonfinite scores,
invalid candidate sets or an inconsistent argmax leave p/score/margin invalid while preserving
those echoes. Such queries count as `invalid_queries`; cycles with either endpoint invalid are
excluded from Ck as `invalid_cycles`. Both depth bits are retained (bit0=depth1, bit1=depth2).
Ordinary Dpc records keep `signal_valid:false` and the actual verification outcome. A mismatched
echo is still a binding failure: only MISSING depths have the -1/-1 sentinel, and these controls
always query at least2 rows. Actual request/card/ctx/full-ID checks remain fail-fast.

## Logging and measurements

Startup: `DSpark matched early verify: on k=K ... query_rows=2 ... logging=quiet|capture`.
Shutdown: `DSpark matched early totals:` and THREE `DSpark matched early costs: k=K ...` lines,
with explicit `[3]` count, cycle_ns, wait_ns, accepted and emitted totals (unused actions have n0).
Steady counters do not depend on debug-ID or JSON logs.

By default Dme independently enables the same read-only observer and copies, but no proposal/cycle
JSON; use ordinary info logging for quiet cost arms. Existing `-v` / `RADIANCE_LOG=debug` still
adds proposal-ID lines and costs time. `Dpc` can accompany ALL three controls for a matched verbose
audit: it adds ordinary proposal/outcome events plus `DSpark matched early cycle:` JSON containing
the earlier seq/epoch/k, verifying target step, next epoch, costs, steady flag and commit status.
It also labels `proposal_invalid_depths` and `next_invalid_depths`; quiet totals count invalid
queries/cycles without introducing per-step log writes.
Join that key to the ordinary capture record for IDs/position/signals. All output is in
`<window>/<arm>.server.log`; no new trace file or service endpoint is introduced.

Price verbose and quiet profiles separately. Log-on C0/C1/C2 include differing numbers of JSON
outcome/proposal writes, debug mode includes proposal-ID formatting, and both can alter overlap.
Do not substitute late-readback Ds2, plain decode, Ds0, or old128GB/pre-sync costs for these Ck.
The early wait itself loses overlap and must be charged. Fixed controls visit different later
anchors because verification emits different counts; compare exact replies/IDs against the target
reference and compare the SAME-profile k2 early/late pair for instrumentation immutability.

## Integration queue and gates

Source cases added to `sched_test`: `dspark_matched_early_controls_parse_and_preserve_configured_query_depth`,
`dspark_matched_early_meter_excludes_transitions_caps_end_and_cancel`,
`dspark_matched_early_invalid_numeric_signals_preserve_ids_and_fixed_prefixes`,
`dspark_matched_early_normal_planning_keeps_the_cap_and_drops_context_end_queries`, and
`dspark_matched_early_db2_keeps_query_and_ahead_publication_prefix_consistent`.
`ds41_dspark_defer_arch_test` adds
`ds41_dspark_early_fixed_controls_observe_the_same_db2_query_even_at_zero`.
The latter records the actual two-row plugin pass at all three k, selector-before-observer ordering,
shared operands, persistent observation shape, and no declaration with both instruments off.

The integration owner rebuilds all plugins coherently under work/AGENT-LOAD-RULES.md, then runs the
focused host/architecture cases and existing probability/issue-ahead controls under build.lock,
without gpu-labelled ctest cases. In a two-lock QFN stop/restore window, retain the observer card
oracle and unchanged base reference. Gate logging-off/on and early-off/on k2, then fixed0/1/2
on code/chat: every reply and generated target token count must match; inspect same-profile k2
proposal IDs and actual query/wait counters, terminal/cancel handling and nonzero steady samples.
Use a small matched verbose pair/triple for epoch audits, then quiet fixed0/1/2 for costs before
any policy-performance inference. Preserve the current192GiB profile as the main measurement and
keep a small-cache regression in the integration queue; no hardcoded cache capacity was added.

Pass measured per-family Ck means and exact source/profile/logging provenance to
[DSPARK-PROB-ANALYSIS.md](DSPARK-PROB-ANALYSIS.md). Keep the frozen request/group split, fit on
training groups only, and evaluate held-out groups once. Candidate softmax thresholds must be
calibrated empirically; historical0.49/0.70 are acceptance break-evens. A later default-off policy
must use the same prefix/lifetime path and be gated on fresh on-policy requests. None is enabled
or implemented by these controls.
