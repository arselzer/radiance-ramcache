# Frozen prefix policy beside concurrency and bank lending

Source only, off35cab28d. No build, syntax check, test, card or model invocation has run for this
change. The first static-bank/lending measurement and its work/bank-lend runner are unchanged.
No configuration is recommended or automatically enabled by this source.

The frozen GLOBAL config remains `v1:0.937915564:0.97501117`; no fitting or threshold selection
is part of this work. Candidate share remains the existing Markov candidate-set share, not a
target acceptance probability. The native two-row Db2 query, original five-row/depth5
declaration, observer, selection arithmetic, StepSlots, bank sizes and target verification do
not change. Only the published/verified prefix remains the policy's existing0/1/2 decision.

## Narrow additional admission

The original max-sequences1 profile, config parser and successful startup line are unchanged.
The new profile admits max-sequences>1 only with the existing native TP4 bank validated on every
rank, deterministic fixed placement, Db2/from1/full5/depth5, explicit deferred context (Dx),
no TBO/tree, no LoRA/media. It does not turn on Dx, dsolo or bank lending. Existing feature
conflicts (CONF, load gate, acceptance gate, compute-off, fixed caps/measurements) remain refused.

Before serving, Engine checks the actual initialized Scheduler dsolo and Dx state. A separate
`DSpark prefix policy concurrent:` JSON line records that proof, sequence bound, native bank,
hotN and exact query shape. Single-sequence startup has no added marker. Bank lending still
requires its own original budget/backup/ownership scope and strict card gate; this change does
not weaken any lender refusal or use its private transaction state as query authority.

Scheduler captures a distinct immutable frame ID at each concurrent-policy block_begin,
together with the existing sole-drafter decision. `policy_block_owner` requires the current
open target step, no issued-ahead frame, and exactly0 or1 actual query owner. A positive owner
must be the captured sequence/slot/card, a non-mirror Db2 row and outside a solo-held frame.
Engine checks it before any context/query or bank mutation and again before observing StepSlots.
Unknown, stale, multiple-owner or reused-frame bindings return an error. Cancellation keeps
the old captured Request protected through block_end; it never reassigns a query to a newly
reused slot. Normal terminal/unpublished outcomes clear the pending meter and publish no new
usable prefix, as before.

## Timing across a concurrent gap

The previous meter cleared no-query target steps, but an idle iteration did not visit that
boundary. The concurrent path now calls its pause boundary for both no-query and idle frames.
It clears pending cost and logs the actual prior sequence/query epoch and next solo span.
The first later completed query logs resume bound to its new epoch/frame/slot/context; it has
no previous cost epoch and zero cycle duration. Thus catching up deferred context and restoring
a bank remain in request/end-to-end wall time, while a multi-request gap cannot become one
"pure decode" sample.

Even if a pause hook were missed, the concurrent meter independently requires consecutive
target steps and Scheduler frames, the same nonzero solo span, sequence/card/owner slot,
previous proposal/context binding and exact output increment. The original single-sequence
meter retains its old admission and semantics. Concurrent acceptance is looked up by the actual
query sequence/card/slot in StepOut, rather than assuming its owner is entry0.

Actions/cycles gain frame/span fields only for the concurrent mode; old S1 rows retain their
field order/content. `pause` and `resume` are additional concurrent-only markers. A numeric
invalid still uses full-prefix2 under the existing rule; no changed reduction, sampler or
query arithmetic is introduced.

## Prepared gates

New C++ source cases cover separate scope positives/refusals; busy/idle/missing-frame/span/
owner reuse never counted as a cost; actual before-commit four-request suppression then the
same anchor's query resume; wrong step/sequence/slot/multiple-owner guards; cancellation and
stale-frame refusal. Existing S1 policy/measurement/issue-ahead cases remain required. Build
and run exact `draft_policy_test` and `sched_test` under build.lock, nice19 and `-LE gpu`.

The strict policy gate retains its default S1 interface. Concurrent use supplies `--nseq 4`,
explicit native-bank/Dx arm proofs and no TBO. It
requires the actual Scheduler startup marker; complete receipts bind every query/outcome,
frame/span and pause/resume, reject a stale pause epoch, and forbid pricing the first resumed
query or a nonconsecutive owner. The A/B source passes actual sequence bounds and refuses a
concurrent Dpp arm without Dx. Fixed Dme controls remain sequence1 only.

The mixed client has one explicit optional `--policy-config` mode; absent it, the existing
bank probe's positive outstanding-draft proof is unchanged. Adaptive k0 can legitimately run
a Db2 query while exposing no verified draft, so this explicit mode instead requires a
no-query pause followed by a live, published, actual anchor query/action/outcome, including
k0. It still requires observed N1→N4→N1, the full32-token solo tail, both window guards,
no hidden warmup, ordered request/sequence mapping and normal full drain. The selected resumed
query context must exceed the anchor progress actually observed while four requests ran;
an earlier unrelated pause/resume cannot satisfy the mixed receipt. Lender-on still
requires every original all-rank lend/restore/copy/mask/hash/resume marker. The new option
does not add, alter or reframe prompts and is not passed by the first bank runner.

Prepared Python source cases: existing policy gate3 plus concurrent3 =6; existing bank probe9
plus adaptive-k0/live-filter case =10; new `dspark_policy_mixed_gate_test.py`6. These are unexecuted.
Run only serialized with the build lock; the fake cases perform no socket/device operations.

## Later root mixed window

Use one frozen workload across no-bank reference, fixed+Dx/lender-off, adaptive+Dx/lender-off,
fixed+Dx/lender-on and adaptive+Dx/lender-on. Fixed role retains its original late cap2 and Db2
query. Adaptive role uses bare Ds/Db2/from1 plus Dpp, explicit Dx, the exact frozen config and
unchanged DvN/Df/hotset. Enable read-only Dpc on all four bank roles for this separate identity
gate. Keep no-bank free of proposal capture. Use the same native target/caches and all other
knobs; repeat96GiB functionally. All GPU/model use remains in the two-lock QFN stop/restore
window, with one coherent full core/all5plugins build and current-library card receipts.

The same mixed probe can run each role; adaptive roles add `--policy-config FROZEN --server-log
LOG`. After successful process shutdown, run `dspark_policy_mixed_gate.py` with --config and
each of --base/--fixed-off/--adaptive-off/--fixed-on/--adaptive-on plus corresponding -log paths.
It requires all five complete receipts, exact same frozen framed bodies/names/prompt IDs,
four UTF8 reply bytes and prompt/generated counts equal to no-bank, and full final policy and
lender proofs. It runs the existing strict capture pairing for every actual query/outcome.
Every adaptive action must bind its own capture epoch/context/anchor and both original IDs.

Cross-role proposals are compared only at actual common (request name, anchor position,
anchor token) keys; query epochs/steps or proposal ordinals across different policies are not
interchangeable. Common anchor coverage must include both initial solo and resumed tail.
Adaptive-off/on also compare at actual common anchors. No proposal equality is demanded of
no-bank. For lender-on, no action may occur between lend-begin and restore-ready, and the
policy's selected anchor resume must be the actual post-restore bank query step.

Literal offline marker:

```
DS41_POLICY_MIXED_BINDING_PASS roles=5 replies=4 anchor_resumed=1 actual_capture_binding=1
```

This is an identity/ownership receipt, not a speed result. Root still binds binary/plugin/
helper/workload hashes, fresh card receipts, startup memory, all load/drain/restore/teardown
wall time, inner success and outer restored-QFN completion. Neither a positive cost tally
nor solo plateau rates substitute for the complete mixed lifecycle comparison. Do not alter
recommended config or claim composition gains until that window passes.
