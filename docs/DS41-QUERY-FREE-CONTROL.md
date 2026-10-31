# Matched fixed0 query-free measurement (source only)

Status: prepared source, unbuilt and unexecuted. This is a cost control, not a predictor,
admission policy or performance result. It must not delay the separate packed/growing-KV cohort.
The earlier historical/feasibility audit is commit b7077af2, `DS41-QUERY-ADMISSION-AUDIT.md`.
No thresholds are changed and no held-out cases are selected, fitted or replayed here.

The separate `DSPARK_QUERY_COPY_STATS=1` measurement instrumentation requires the same native
fixed0/Dx1/Db2/N1/greedy profile in both Q and NQ. It counts successful observer and StepSlot D2H
enqueue calls and their actual byte arguments at the two existing rank0 query-copy sites. A real
pre-epoch step/seq/slot/card owner is bound to the normal query epoch only AFTER existing context
and observation-echo ownership validation. No target copy, timer, copy or device operation is
added. Its six pure host source cases cover successes, failed enqueues, orphan/duplicate/wrong
width copies, pending-frame refusal, owner/epoch mismatch and empty NQ state; still unexecuted.
Existing totals append rank0 scope, calls/bytes, actual physical widths, bound epochs and final
unbound/mismatch counts only with this explicit flag. NQ must have no pending frame/epoch/copy;
Q requires exactly one successful observer/slot pair per actual bound epoch. Flag0 preserves
previous summary text. The new SOURCE runner in `work/autonomous-20261031/query-free-cost` uses
the explicit paired flag, actual origin metrics and normal guarded lifecycle; the old default
Dmnq harness guard intentionally does not accept this richer receipt without its dedicated gate.

## Exact scope and seam

`RADIANCE_EXPERIMENTAL_DS41_DSPARK_QUERY_FREE=1` requires explicit fixed early0 and Dx1.
Only the native TP4 deterministic fixed DSpark bank, Db2/from1/full5/depth5 and max-seqs1 are
accepted. Tree, TBO, dynamic Plan, media, LoRA, bank lending, compute-off, verify-cap,
acceptance/confidence/load gates, probability capture and adaptive prefix policy are refused.
Every admitted request must actually be greedy (finite temperature and `sp.greedy()`), with no
XTC draw, grammar/lazy grammar, thinking budget, control vector, ngram ban, logprobs, embedding,
scored request, plugin words, media or LoRA. This uses actual request parameters rather than
assuming `--deterministic` removes sampling RNG. Unsupported requests are refused before a slot
is allocated. Target sampling parameters are never rewritten.

The declared head, observer buffers and pinned observation allocation remain unchanged. So do
target `ADV_ACCEPT`, the existing sampled-ID `collect_issue` readback, fixed0 target pipeline,
normal `block_begin` eligibility and its full Db2 KV reservations, every Dx context pass,
snapshot save and `block_end` filled-range update. The new branch returns after context and
snapshot save, before `block_query`, `ADV_BLOCK`, query `block_pass`, `ADV_BTAKE`, confidence/
probability issue and query StepSlot readback. No proposal is generated or published.

An eligible NQ opportunity requires a real completed context pass and the actual sampler's
`speculation_fixed` proof. The context pass already executes `build_greedy(0)` on every rank;
the omitted query's second reset is idempotent. A zero-context opportunity is refused, rather
than quietly dropping a reset or adding a differently timed reset. Native target sampling still
starts with its ordinary `build_step`; request RNG coordinates and inflight ownership are untouched.

Before a multipart prompt produces its first token, `Engine::block_shape` still returns the
declared full5/depth5: `decoded_seqs()` counts producing plan entries, and Db2 starts on the
last prompt piece. Such a full5 frame is accepted only as an empty pause (`n=0` with no
`can_draft` owner). It creates no opportunity, boundary, wait or query epoch, and leaves Dx tap
history owed. The first producing Db2 piece must run all retained context before it can form an
opportunity. This corrects the first M16 piece of the 28-token TRAIN request failing before
generation; it does not alter the query controls or promote empty full5 frames to measurements.
Any nonempty full5 frame, unknown/reset shape, stale phase/epoch, mismatched owner or incomplete
context remains refused. Error paths now identify `stage=opportunity|ready|finish|block_pass`
and a static `reason`; successful paths add no per-step logging.

At fixed0, `ADV_BLOCK` only stages query inputs and does not mutate target StepSlot state.
`ADV_BTAKE` only fills proposal tokens (and tree fields, excluded here). The next target
`ADV_TRUNK` reads `tok[0]` because its speculative depth is0. The retained `ADV_ACCEPT` supplies
that target token/context. Old query writes to speculative drafter KV are outside the filled
range; the next retained context pass overwrites those locations before their normal use.
No target/compressor/Engram arithmetic, kernel, shape, checkpoint format, placement, cache
budget or model ABI is changed. Flag0 follows the previous branches and log lines.

## Separate measurement, without fake query epochs

NQ still waits on the **same existing `step_end_` event before `issue_ahead`**, only when normal
fixed0 would have an eligible query. It reads the existing target-ID readback; no new device
event or device copy is introduced. It has its own monotonically increasing opportunity
boundary, never calls `draft_prob_reserve_epoch`, and never manufactures probability outcomes,
proposal IDs or `solo_held` evidence. Query dispatch and observation readback explicitly fail
if reached in this mode. Each opportunity/commit additionally requires the query-epoch counter0.

The candidate is captured under scheduler ownership after all queued context rows finish and
before ahead planning. It binds actual seq/slot/card, current step, input/expected committed
context and output count. After the normal target commit, its token, exact +1 output and
committed context must match the same live request; a cancelled/finished/reused owner cannot
validate it. The meter closes only consecutive pure-decode opportunities of the same owner,
step+1/boundary+1, continuous context/output counts and increasing time. Prefill transitions,
first decode, no-opportunity gaps, caps, completion, cancellation, stale frames and slot reuse
cannot be counted as one pure cycle. This is `C0NQ`, distinct from existing Q `C0`.

Startup, one actual normal-path first-use row, shutdown totals and one `C0NQ` cost row form
the receipt. There is no per-cycle INFO logging. The strict gate requires actual context/use,
zero query/issues/observer/epochs/publications, nonzero consecutive cost coverage, exact
opportunity classification and matching integer cycle/mean sums. End-to-end request and process
timings still include all ordinary load, context, target, snapshots, drain and exit work. The
steady-cycle difference does not measure deferred-context savings or a future admission policy.

Prepared checks (all still unexecuted):

- `draft_query_free_test`: six pure cases for parsing, scope/request refusals, actual-commit clock,
  prefill, gap, owner/slot/card reuse, cancellation and context/time/output discontinuities.
- `sched_test dspark_query_free`: six real scheduler fake-backend cases comparing Q0/NQ retained
  Dx rows/positions and target-ahead tokens; refusing incomplete context/stale step/nonzero proposal
  epoch; actual cancellation and replacement-owner exclusion. The multipart case follows the
  actual engine full5-to-Db2 shape switch over 28- and 80-token prompts at M16, requires one/four
  initial empty pauses, exact complete 16+12/five-by16 context catch-up, subsequent one-row
  context, fixed0Q/NQ output and ahead-token equality. The companion case refuses nonempty
  full5, unknown/reset shapes, omitted owners and stale/malformed frames. No model or emulator
  involved. These added cases are prepared source, not an executed proof.
- `python3 -I tools/ds41-host-experts/test_dspark_query_free_gate.py`: eight fake receipt cases,
  missing/duplicate first use, wrong owner/context, nonzero query/epoch/publication and wrong sums.

Root must build/run these under `work/build.lock` and load rules; this lane ran no build,
syntax check, test, replay, fitting, emulator, model or GPU command.

## TRAIN-only fresh matched window recipe

Use two separately bounded windows, code and chat, six roles each. Optional paired reverse-order
repeat is predeclared before inspecting results. A later96GiB functional pair is independent;
the control itself has no hard-coded192GiB budget. Preserve the existing framed completions,
prompt sets, tokenization, names, order and caps. There is no warmup or extra hidden generation.

Only the frozen TRAIN receipts from
`work/autonomous-20261031/dme-cost/training-receipts.json` are allowed:
`Ucodecost256` (13) and `Uchatcost` (12). Before launch, bind that file, both exact receipt files
and their source prompt files by SHA256, checking names/order/counts against the frozen receipt
inventory. Code uses original `hot-both-db2.json`, chat original `hot-heldout-db2.json` (the latter
is the historical **hotset filename**, not permission to use held-out measurement requests).
Do not select different prompts/caps after a result. Leave existing sets untouched.

Use the historical quiet cost profile with explicit N1 and Dx on **all bank roles**, including
fresh 0Q/1Q/2Q; old quiet costs without Dx are not this matched control. Example arm components:

```sh
P=all-t40-f-M16-H-p256-m196608-G-R-B1-W-A2-Rd13-Er-Es-Pa-K-Sc2-F8-Pp4-Ei-Pm56-Pe-Rb-Ss-Pu-Is-Pi8-Ns7-Fw-Rm7-Vp-N1
D=$P-Ds-Dkt-Db2-Dv24-Pmr-Df-Dx
# F is codecost256 or chatcost, frozen before either run.
plain=$P-U$F
q0=$D-Dme0-U$F
nq0=$D-Dme0-Dmnq-U$F
q1=$D-Dme1-U$F
q2=$D-Dme2-U$F
v1=$D-Dpp-U$F
```

`Dmnq` requires `Ds-Dme0-Dx-Db2/N1` and refuses Dpc/Dpp/Tb/gates. Every harness arm sets
QUERY_FREE explicitly0/1. Only the NQ role has its separate startup line and gate; it cannot
pass the old matched-query startup guard by claiming it queried. Frozen V1 remains exactly
`DSPARK_POLICY_CONFIG=v1:0.937915564:0.97501117`, with its existing mandatory policy logging cost.
Set RADIANCE_LOG=info, no Dpc, max-seqs1 and the same deterministic/prompt/cache/runtime knobs
for every applicable role. Preserve raw BF16/F32 ordinary memory paths for this independent gate;
do not silently add packed KV/window, B16/compact index, growth or other unproved memory modes.
No new bank KLD arm is proposed: the native fixed-bank scope deliberately refuses KLD.

Root snapshots this branch's entire tools directory (the new gate must accompany ab_window),
the source commit, actual serving binary and all five plugin hashes from one coherent build;
both homes must be home-sync. Use the existing `launch_window.sh`/chain guard under both research
locks, stop/restore QFN, and required fresh native draft/Rb/Vp card receipts bound to the exact
library. Recheck ordinary QFN restore and answer before accepting any analysis. No service or
GPU work is authorized outside that guarded window by this document.

Predeclare six role order and immutable per-family manifests. Require all requests complete;
same names, actual prompt IDs, full UTF-8 texts, prompt/generated counts and caps against both
fresh plain and Q0 reference. Validate each request's exact seq mapping (no warmup), all old
Q0/1/2 query/observer/context and fixed-prefix gates, frozen-V1 epoch/action ownership, and NQ
startup/complete `QUERY_FREE_GATE PASS`. Compare Q0/NQ actual context issue/row/position evidence
where the existing observer/capture can provide it without a new differently timed copy.
No proposal-ID equality is expected from NQ because no proposal exists. Require zero actual
NQ proposal epochs and no unsupported-path warning/fault. The host target-ahead proof is required
before the model role; it does not replace byte-identical model responses.

Compute weighted TRAIN `sum(cycle_ns)/sum(n)` separately for Q0 and NQ0 and report counts,
exclusions, complete request decode+total rates, TTFT sums and total lifecycle including load/
drain/exit. `delta=C0Q-C0NQ` is the first matched query+observer/readback cost estimate. Do not
use wait time alone, force a positive delta, mix old Dx-off costs, infer a held-out gain, or
claim a live predictor from this control. Any later lagged-signal admission experiment needs a
new authorization/gate with its own context, missed-yield and bank-lifecycle contract.
