# Read-only evidence for a possible Phase2 executor

Source only. This adds no executor, retention, expert union dispatch, altered routing, expert
ordering, arithmetic or cache policy. No build, test, syntax check, device run or model run has
been performed for this change. The unused route-frame prerequisite remains separate.

`RADIANCE_EXPERIMENTAL_DS41_P2_AUDIT` is off when unset, empty or `0`. An absolute output prefix
enables the instrument, for example `/absolute/window-dir/p2-routes`. Each rank exclusively creates
`<prefix>.rank<r>.jsonl`; an existing file refuses startup. Quotes, backslashes, control characters,
relative prefixes and prefixes longer than1024bytes are refused. The initial runtime scope requires
the actual static native DeepSeek41 EP4 plan,40layers,384experts,top6,5120hidden, admitted LCP4096
and P1. Early-issue and two-batch plans are refused. An unavailable native stager refuses startup,
including architectures which would otherwise return before initializing a file stager. Default-off
runs allocate no trace, file or readback, issue no audit copy/event and emit no audit marker.

The audit must be enabled only in a normal two-lock QFN stop/restore window. It is intentionally
excluded from performance comparisons. JSON formatting, disk output, extra device reads and the
extended existing counts wait all cost time and can perturb asynchronous drive arrival timing.
The original scheduler, read-ahead, ring scheduling, weight selection and launches remain unchanged.
Any later speed comparison must explicitly set this variable to `0` for every performance arm.

## Identity and ownership

`Engine::prepare_release` copies the actual host `StepEntry` before releasing the parked ranks.
It accepts only a single pure-prefill entry with no device-advanced rows, draft/speculation or
encoder, matching the batch's sequence list and token count. It records scheduler `seq`, card slot,
absolute original-piece context position, query length and prompt end. Ahead planning in the
current scheduler creates decode-only plans, which are excluded. A missing/mixed identity never
borrows the previous request's identity.

The successful EP scatter witness retains its actual current IDs, sorted and offset operands only
when requested. The existing producer whitelist, native dimensions, local96 numbering, no protected
expert permutation, compute lane and generation checks remain authoritative. Additional audit
shape checks require dense I32 `[rows,6]`, `[rows*6]` and `[97]`. Unsupported shapes leave optional
pointers null and produce `producer_operands_unavailable`; no old GU pointer is substituted.

At `start_ep_routed`, the audit queues IDs before the existing counts completion event. It reuses
the existing P1 admission (or CPU cut-layer fetch) sorted/offset copies **only when both source
pointers exactly match the current successfully issued producer**. Otherwise it queues sorted and offsets into its own
bounded pinned staging. All copies use the same producer stream, before the same counts event.
After the existing successful wait and counts validation, `routes::freeze` validates/copies the
actual arrays before staging is reused. The native local96 count row is always reused; there is
no extra count read. A foreign-only or dropped `-1` row is valid and invents no local expert.
The sorted dead tail is retained verbatim, but only the live local prefix is interpreted.

Original FFN suffix rebasing is preserved: IDs are already sliced at `row_from`, while sorted
slots, offsets and counts start at zero. Both absolute original-piece `begin/query_rows` and
`row_from` are recorded. Each piece retains its **actual Ctx generation** as `epoch`; `span_epoch`
is merely the first such generation used to group contiguous observed P1 pieces. Nothing retags
an engine frame. Producer serial, step, sequence, card, layer, rank, Program and Plan cookies are
included. Cookies identify the actual process-local objects; they are not binary/source hashes.
The window manifest must independently bind source, binary, all plugins, full library SHA,
configuration, prompt IDs and output hashes. Logs alone do not authorize a configuration.

FileStager owns one pinned staging allocation: exactly
`(2*24576+97)*4 = 196996` requested bytes per rank. On AQL's no-SVM pinned fallback, its conservative
host-allocation charge is4MiB per rank (`ceil(request/2MiB)*2MiB+2MiB`), rather than the requested
payload alone. The fixed `Trace` object contains one frozen frame and40×96 bounded unit summaries;
startup prints its actual `sizeof`. No full-span activation, model data, device history, event queue,
background writer or per-frame retained vector is allocated. Freeze additionally uses its fixed
temporary Frame and24,576-byte validation bitmap on the issuing thread's stack. Existing counts
and admission staging remain owned by their original paths.

Every route records the actual `extra_readback_bytes`, exact order-reuse choice, and the existing
measured `counts_wait_ns`. Added payload is `rows*6*4` for IDs, plus the same number for sorted
slots and388bytes of offsets when order cannot be reused. At4096live rows this is98,304bytes with
reused order or196,996bytes without, per layer/rank. `counts_wait_ns` includes the original router
and counts dependencies as well as the added reads; it is not a separately measured audit-only
latency. It does not include later host freeze/JSON/disk time. A matched off/on diagnostic can price
total overhead, but neither its timing nor subtracting two waits is a feature-speed measurement.

Successful readbacks are consumed only after the counts event, before reuse. On an aborted issue,
the instrument stops accepting records, leaves its staging alive, and relies on the same Rank
ownership as `counts_host_`: Ctx shutdown drains streams before the FileStager is destroyed. It
must not use a remembered stream in FileStager destruction, since Ctx already destroyed it.
No instrument error changes the normal fixed reply path. Explicit invalid startup configuration
refuses startup; runtime incomplete evidence poisons only the diagnostic.

## Actual copies and classifications

At each actual layer, the trace builds a complete96-entry map from the immutable Plan, including
actual movement-unit IDs, tier and native unit byte count. Effective resident means static VRAM,
`Mover::ring_resident`, or an actual file unit in the slab at that instant; other admitted P1 sources
are nonresident. It does not infer GPU residency from the RAM hit rate, profile share or cache size.
All96 entries are recorded, including unselected and resident experts.

`CardExpertStager::queue_one` calls an optional read-only observer **after** the unchanged device
copy returned success and the original counters/state were updated. The record contains actual
local/global expert, Plan unit, routed rows, ring slot, source kind (`0` RAM lease, `1` drive ready
slot, `2` pinned pool), packed/native flag and the actual GU/DN DMA lengths. It does not count
read-ahead which never reached a ring copy, slab residents, fixed VRAM residents or a failed enqueue.
The observer retains no Source pointer or lease and returns no scheduling decision.

`enqueue_succeeded` is not an independent per-copy completion claim. A P1 layer closes only after
the original `run` issued its waves, with one actual copy for every selected nonresident expert.
The original P1 end-pass source/event drain precedes `piece_end`. The normal engine's returned
step error subsequently poisons the audit even if a final route record was already complete.
Logical cut/key-only layers are explicitly recorded without inventing route IDs or copy events;
a missing live layer, duplicate frame/copy, stale generation, Plan alias, wrong ID/row binding,
unfinished P1 layer, or missing selected nonresident copy is incomplete.

## Coverage and projected byte bounds

Each analysis span contains at most4 contiguous original P1 pieces and8192queryrows for one
sequence/Plan/rank, preserving40 independent layers. A sequence/position boundary, non-P1 pass,
size boundary or shutdown closes the covered span. Short tail/sampler passes and decode have
explicit `excluded` records. An armed P1 pass with multiple prefill sequences or an unbound identity
poisons the diagnostic with `unbound_p1_identity`, rather than silently omitting actual P1 work.
This is coverage of
the observed P1 span, not automatically the whole prompt: for the6817-token case, the original
two P1 pieces cover6816 prefillrows and a final short pass remains outside that span. Prefix resume
may likewise start above position0. `whole_prompt_covered` states this distinction explicitly.
Cancellation does not erase actually issued work. Records make no request-commit/accepted-output
claim; `piece_end` explicitly states `coverage_only=true,request_commit_proven=false`. Request
completion/cancellation and identical replies require the normal external window receipts.

For each layer/unit, the ledger records selected nonresident occurrences separately from actual
ring copies. CPU-path cut suffixes can have selected nonresident experts with no ring copies and
are not priced as transfers. Across the covered pieces:

* `copy_bytes` sums the actual successful enqueue GU+DN lengths.
* `duplicate_nonresident_same_representation_bytes` sums later copies of the same immutable unit
  with an already observed identical `(packed,GU-length,DN-length)` tuple. Mixed formats remain
  explicitly counted; repeat tuples inside a mixed unit are still counted exactly.
* `union_min/max_observed_representation_bytes` sum the smallest/largest actual copied representation
  per unique layer/unit. `projected_repeat_bytes_min/max` subtract these from actual copy bytes.
  These are **observed-representation accounting bounds**, not a proved union executor: retaining
  one representation for every future reader, window/state ownership, capture lifetime and memory
  admission remain unimplemented. There is no speed forecast or guaranteed avoidable traffic.

Source epochs are checked before capture; the same Plan unit/layout refers to immutable source
weights, not newly requantized alternatives. The instrument does not scan/hash model weights.

Maximum output is64MiB/rank,512 actual route frames and16 spans for the process. Each full frame
reserves a conservative textual byte bound plus closure/copy metadata before accepting it; no
unbounded queue or malloc growth follows prompt length. A cap emits `DSP2AUDIT incomplete ...`
and poisons further recording. It does not silently produce a complete truncated receipt.

The exact stdout families are:

* `DSP2AUDIT enabled rank=... file=... program=... plan=... ... performance_eligible=0 executor=0`
* `DSP2AUDIT span rank=... seq=... span_epoch=... ... complete=1 ... projected_only=1 saving_claim=0`
* `DSP2AUDIT incomplete rank=... step=... seq=... epoch=... reason=... performance_eligible=0`
* `DSP2AUDIT refused rank=... reason=...` (startup failure)
* `DSP2AUDIT end rank=... complete=0|1 frames=... spans=... performance_eligible=0 executor=0`

A valid diagnostic requires exactly one enabled and terminal complete1 line for each rank0..3,
nonzero frames, zero refused/incomplete lines, ordinary successful server exit, valid complete
JSONL/footer and all-rank matching step/sequence/positions/generations for each covered piece.
Duplicate/stale/missing rank receipts are not valid. Span messages alone are insufficient: a later
engine/output failure or truncated shutdown invalidates the process. Limits are diagnostic
admission bounds, not context or cache-budget constants;192GiB and96GiB expert budgets remain
independent and untouched.

## Required first verification (integration owner only)

Build coherently under `work/build.lock`, then host-only `ds41_phase2_audit_test` and
`ds41_phase2_routes_test`, with `ctest -LE gpu -R '^ds41_phase2_audit_test$'` / the other exact name.
Nine prepared cases cover the actual host-plan/sequence binding, generations/seq0/unit0, immutable readback reuse, resident exclusion,
mixed and repeated formats, foreign/dropped/key-only/CPU rows, wrong units/aliases, stale epochs,
missing/duplicate copies, aborted/short/unbound pieces, caps, all ranks and successful-versus-failed
actual scheduler copy notifications. These have not been executed.

Then one normal two-lock window: existing4K P1 reference with audit0, an otherwise identical arm
with a fresh audit prefix on one6817/8K prompt, plus a96GiB functional repeat if desired. Keep
existing prompt sets unchanged. Require complete reply bytes/counts/prompt IDs/Kh equal the
reference, all4 clean terminal audit receipts, exact native frame/copy binding and output/footer
parse. Independently reconcile summed actual copy counts/bytes to the existing P1 counter deltas
over exactly the covered pieces; excluded short/decode passes must be identified. Do not compare
audit-on timing as a performance result. Price the extra bytes and existing counts-wait extension
separately; a follow-up executor proposal needs actual duplicate units before any saving claim.
