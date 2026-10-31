# Phase2 executor prerequisites after the4K gate

Source-only, off44f50fb4. Nothing here has been built, syntax-checked, tested, simulated, run on a
model/card or enabled. No Engine, Scheduler, arch, P1, runtime flag or runner calls the new helper.
The new source stage is `core/place/ds41_phase2_routes.h` plus six prepared host cases. Existing
`ds41_phase2_retention.h` remains a separate unused accounting helper, not executor admission.

The independent4K model gate already exceeds the original500tok/s target on the3086/6817-token
documents. For6817 tokens the measured2K→4K P1 traffic is391.08→254.80GB and21724→14154 ring copies,
not a prediction for Phase2. Neither the remaining254.8GB nor every repeated layer is necessarily
redundant: only actual per-piece expert-ID unions can establish what a span could avoid copying.
Current priority is concurrent decode, so any persistent resident displacement must be priced.

## What was already tried

The prior full archive audit and fresh relevant-source/design reads cover Phase0/1/2 proposals,
PREFILL-200-DESIGN including its critique, PREFILL-DESIGN/CRITIQUE, PREFILL-REPLAY and PHASE1-PLAN.
Phase0's early DMA/card0/root-complex repeats include watchdog-contaminated runs; do not reuse
their absolute timings as clean evidence. Early unbalanced LCP lost2.2–3.4× from dead tails/skew;
live-tail, balanced twins and P1 subsequently fixed that baseline. LCP, P1 ring/packed supply,
causal last128 admission, end alignment, arena lending, consumer cuts and4K pieces already exist.
Whole-unit read-ahead, cache-policy alternatives, TBO, ring/grace retries and more RAM capacity
have prior negative/neutral evidence. They do not remove a second piece's actual expert DMA.
Phase2 layer-major execution has not been implemented/measured. Old80/122KiB/row plans omit live
planes; the current conservative8K-context inventory is176446B/row (172.3105KiB),1147.11MiB/card
for6817 rows, before metadata, window/KV and captures. No old forecast is a current gain estimate.

## New closed metadata stage

`routes::freeze` takes one ORIGINAL piece's global top-six IDs, local96 counts,97 offsets and
piece-local sorted array. It copies them into a fixed-size value object with no allocation or
borrowed pointer. Binding includes sequence, a common span epoch, each piece's ORIGINAL frame epoch,
immutable Plan cookie, layer, logical rank, original piece ordinal/absolute start/query height and FFN row suffix. Sequence0,
expert0 and Plan unit0 are valid values; repeated ID−1 preserves original unselected slots,
not expert0. Epoch/cookie0 are unbound. Ordinary pieces can have different actual frame epochs;
they are retained separately and never overwritten with a shared epoch merely to make a union.
Only native pure prefill is accepted. Decode/draft, wrong shapes/ranges, duplicate top-k choices, invalid prefix sums, missing/
duplicate/wrong-owner sorted slots and stale/mixed span bindings refuse without changing output.

The crucial current shape contract comes from `MoeFP8::pass_head`: IDs/weights and FFN input are
rebased by r0, but sort/offset/count and expert down outputs start at0 for the pass's `rows*6`
slots. Only the local prefix `offset[96]` is live. The tail holds other ranks' dropped slots;
its bytes are copied verbatim and never interpreted as extra expert rows. An explicit key-only
piece has zero FFN rows and all-zero local counts/offsets. A positive piece selecting only foreign
experts likewise produces no local job; it must not invent expert0 or a fake wave.

`routes::make_union` requires all original pieces in contiguous absolute-position order, at most
four pieces and8192 total query rows. It deduplicates local expert IDs/actual Plan unit IDs while
preserving every piece's original [offset[e],offset[e+1]) range. It does not build a giant sort,
change top-k decisions, return a larger GEMM height or infer a DMA source. Selected unit aliases
and missing residence classifications refuse. Resident and nonresident duplicate occurrences are
counted separately; these are counts of potential reuse, not copied bytes saved. Cross-layer,
sequence/epoch/rank/Plan and reordered/gapped piece unions refuse atomically.

Prepared `ds41_phase2_routes_test` cases cover a literal nontrivial two-piece permutation with an
expert repeated across pieces, input readback reuse after freezing, key-only/foreign-only pieces,
expert/unit0, dead-tail bytes, malformed metadata and atomic refusal, stale/mixed scopes, and the
4096+4096 bound with all96 experts on all four logical ranks. No test was executed. The fixed
value-object size is explicit through `metadata_bytes`; it also charges freeze's temporary frame
and validation bitmap, the union temporary and a small bookkeeping margin. Readback buffers, events,
launch snapshots, library/arena leases, checkpoints
and existing retention/KV costs are additional. This helper grants no allocation/execution rights.

## Exact next instrument, before an executor

Add a separate optional read-only diagnostic after P1's existing routed-count synchronization,
before another router/row slice can overwrite its buffers. Default-off must create no extra
declaration, allocation, issue, readback or log. Do not silently extend oldRt's<=16-row contract.

For each actual original piece/layer/rank, freeze a coherent witness containing the IDs, counts,
offsets and sorted slots from ONE routing generation. IDs are the actual rebased suffix, not the
start of the full query plane. The counts pointer must match the `Ctx` EP producer witness that
`start_ep_routed` synchronized; do not label an asynchronously replaced row with an earlier epoch.
Read the extra arrays under a recorded stream event and wait before reusing host buffers. At the
native maximum this is196608B of IDs+sort plus772B of counts/offsets per piece/rank. Budget its
host/pinned storage and measure off/on overhead. No numerical kernel or placement change is needed.

Use a new versioned receipt, with explicit source/library/profile authority, not inferred config.
Required identity: model/checkpoint, loaded kernel cohort, Plan/residency epoch, request, common span
epoch, actual frame epoch per piece, logical rank, layer, original piece ordinal, absolute begin, query_rows, row_from and rows.
Freeze actual Plan unit identity/geometry and actual resident/P1/CPU classification alongside it.
Pair each P1 copy with that unit and its actual packed/block byte lengths, source kind, copy/free
events and every consuming piece. Record:

- Original expert occurrences, unique experts and duplicate occurrences, resident versus
  nonresident and P1-copy-eligible versus CPU/cut pieces.
- Actual copied unit IDs/bytes per original piece, including packed/native fallbacks; do not
  multiply all duplicate IDs by18.8MB or rename pack counters as relay bytes.
- Frozen frame hashes, missing/stale/invalid frames and complete terminal records. A partial
  trace or an unbound response is not evidence of a union saving. Invalid diagnostics must be
  labelled/discarded without changing or failing the ordinary prefill/reply.

First gate is logging off/on with the exact current4K6817 prompt and one96GiB spill case. Require
identical prompt/generated IDs, replies/counts, hashes and source configuration; then quantify
only the actual repeated nonresident IDs/bytes. The one-request fixture must also match the
observed2736+4080 original card-piece rows (6816 prefill rows, with the final prompt/sampler row
remaining outside this span). Do not derive these pieces with ceil(prompt/4096), and do not fit a
new piece height from aggregate shutdown counters. The helper itself does not run this instrument.

## Smallest eventual active scope

The first executor should be a native, static, pure-prefill span of ONE fresh sequence, at most
8192 unsampled rows, comprising ORIGINAL<=4096 pieces. Require explicit no-prefix-cache and no
requested intermediate/EORS checkpoint, no resume/COW owner, speculation, media, LoRA, concurrent
decode, KV streaming or live-growth transaction. Initial composition keeps packed-KV/window,
BF16-score-plane and other unmeasured format combinations off unless separately gated. Reject
unsupported combinations before any write;
do not silently change the user's prefix/cache settings. All model/kernel/geometry and norm pins
must match the already gated4K cohort. The final normal prompt/sampling row follows the completed
span through today's unchanged path. Shape/eligibility failure selects ordinary4K before starting.

A useful first executor can be **producer-only layer-major**: layers0..19 over all original pieces,
then resume layers20..39 in original piece order from retained L20 inputs. This avoids inventing a
new giant consumer pass, preserves its per-piece key/full cuts and avoids changing the CPU/card
threshold for small consumer suffixes. Producer pieces must all use the existing P1 card path.
It still requires the complete lifetime/state protocol below; this is not an implementation shortcut
around KV/checkpoint safety. Measure producer-only duplicate traffic before expanding scope.

1. **Prepare without publishing.** Under the scheduler lock, derive the exact ordinary4K piece
   list using `plan_chunk` on projected counters, preserving end alignment, quantum, target/cuts
   and checkpoint requests. Do not call normal scheduler commit to obtain the next piece. Hold
   the request and all allocations under a span ticket. The optional span entry needs an explicit
   bounded descriptor/version contract with the architecture; current `rad_arch_step` has only
   one mutable RadBatch. A DS41-only optional export is a possible narrow interface; do not encode
   an opaque span into an existing field or invoke ordinary step repeatedly and pretend it did
   not publish state. Exact missing/export/cohort refusal is required.
2. **Fund and hold the full span first.** Reserve the conservative retained planes plus FULL
   compressed/index pages and full-span WINDOW backing/IDs before writes on any rank. Keep
   window keys' plain BF16 arithmetic and rawF32 compressor/token states. First funding is the
   full retained layout, with `replaceable_piece_bytes=0`, unless actual arena/alias accounting
   independently proves a replacement. Do not subtract the old piece subset just because it is
   modelled in both layouts. A plain window's4K→8K additional working range is about160MiB/card;
   actual page/table alignment and all40 stripes determine the charge. Keep variable192/96GiB
   expert budgets, future unbacked cache and configured headroom. Failure occurs before mutation
   and falls back to4K. No rolling-window saving is admitted without a full pin/read inventory.
3. **Own immutable batch views.** Every original piece gets its own token/absolute-position/cu
   arrays, per-group slot mapping, block table, seqused, state-index row/pitch, source bindings,
   cut rows and output-row metadata. Preserve their old values/causal extents, not a new
   `cu=[0,nf]` guessed from the FFN suffix. `BatchBuilder::Set`, BtMirror, Ctx route counts and
   advance-graph patches are currently reused; a saved pointer into them is not a span view.
   Full private uploads/readbacks must remain alive through their final consumer. No recorded
   advance may rewrite a frame behind the executor. No physical ID can be released/reassigned
   merely because its earlier logical chunk would have ended under the old scheduler.
4. **Keep original state transitions once.** The ordinary step prologue's token-map, embedding
   and initial stream/pre_f seeds run once per original piece before layer0, with no span head/
   sampler result. At each layer, dense pieces run in absolute token
   order at their old call heights and key/full cuts. qsa_work/softmax_pool/entry_store/tail_store
   run exactly once for each original source piece; rawF32 tail order, compressor page outputs,
   partial ratio groups and unwritten padding are unchanged. Compressor `page`, `bpos`, `nc` and
   stage arrays are piece-local until their stores complete. FULL index/compressed pages may
   already contain later pieces when an earlier query runs, but original per-row reach/masks
   must exclude them. Retain index_sel per piece between source layers and layer20 index_keep
   through later index layers. Token-history/Engram state is independently striped per bound
   layer: retain original token order, apply each update once, and do not expose its future
   history as a checkpoint. No Engram dedup/reordered gather is part of the first executor.
5. **Retain and replay actual expert frames.** Split route/shared/resident preparation from
   nonresident supply and gather. `MoeFP8::pass_head` currently interleaves all three arms; Seg1/
   Seg2/Seg3 alone is not enough. The new per-piece launch owner needs independent tensor/param
   arrays, immutable strings or explicit Program/library ownership, exact selected function/row,
   preserved original M, rank/layer/span generation, null instance and scratch-free admission.
   Rebase only explicitly declared retained inputs/outputs and the per-wave weight table.
   Preserve original per-piece offsets/sorted arrays; do not concatenate or re-sort. GU/SwiGLU
   scratch may be shared sequentially across pieces only after each captured down launch uses
   it; expert_down, FFN input, routing/shared/mHC and next-layer norm output remain retained.
   The arena/liveness and actual operand guard must know these retained bindings; RAW pointers
   are not permission to evade bounds or lend a live region to P1.
6. **One supply, every original consumer.** Fetch/lease a unique nonresident unit once, with the
   existing packed-RAM, pinned-pool or actual miss/exception path. Replay its known GU/SwiGLU/DN
   triple on each original piece that selected it, with that piece's counts and sorted ranges.
   A ring slot's free event goes AFTER the last consuming piece, not the first down. Source
   leases/ready slots last through DMA, and device tables/events last through every replay.
   Resident outputs also use their original piece calls. Gather stays in original six-slot
   order, with the same shared contribution and ascending-rank collective. Cache admission is
   a separate explicit policy: retain each piece's causal last128 admission decisions; do not
   silently replace them with future routing/oracle scores or assume every union unit is a hit.
   The resulting admission/recency ordering and its decode impact need a stated/gated policy.
7. **Publish or cancel at one cohort boundary.** The first scope has no shareable prefix or
   intermediate snapshot, so a failed/cancelled fresh request can discard its entire unpublished
   span after every rank's queries, kernels, copies, leases and tables are drained. Stop only at
   a coordinated boundary; one rank leaving while peers enter a collective deadlocks. No
   prefix, n_computed advance, cache checkpoint or sampler result is published midspan. On
   success, all four ranks' completion receipts precede committing the original piece metadata
   in order; invalidate/restage batch mirrors before returning to ordinary decode. GPU/restore
   failure is fatal where exact rollback cannot be proven. Extending to resumed/shared requests
   requires staging each checkpoint's per-layer raw state/window bytes and atomically publishing
   only after all stripes are complete; reject it until that separate transaction is implemented.
   Even with prefix publication disabled, the validation hashes need the ORIGINAL per-piece raw
   state. Capture each overwritten compressor/token-state stripe at its logical piece boundary;
   never hash final-span state repeatedly and label it as earlier checkpoints. Full-span unique
   window/FULL pages can support historical logical views only where their exact written/cut
   ranges remain intact. The hash oracle therefore needs owned snapshot views and the original
   masks/run order; fund and gate that seam before claiming Kh. This is still a runtime blocker.

## Concrete current call sites

- `deepseek41.cpp::step/run_layer` (3488; Seg2 at3891, Seg3 at3921): Seg1 includes dense/KV/FFN
  input; Seg2 calls pass_head, which
  still launches experts/shared; Seg3 calls pass_tail plus next-layer fused post/collapse/norm.
  FR/KR are original piece-relative row slices. The fused next xn is live across the boundary.
- `rad_block_moe_fp8.h::pass_head/pass_tail` (1735/1931): router writes counts into Ctx's one layer row;
  sorted/offset/count and edn are pass-local while IDs/weights/shared input use r0. Preserve this
  distinction in every captured frame and typed retained binding.
- `CardExpertPath::capture` (`card_expert_stager.cpp:770`; Cap atheader454) currently rejects a second triple in one pass and any scratch bytes;
  `Cap` owns t/p arrays but shallow-copies tensor data and parameter string pointers. Existing
  `wave` (`card_expert_stager.cpp:709`) patches ONE table and launches ONE triple; it cannot
  service a multi-piece union yet.
- `KVManager::ensure/recycle_window` (`kv.cpp:1508/1483`, capacity at1463): ordinary ensure recycles from the step's first query, then
  updates one sequence block list. `window_seq_cap`/`window_pool_cap` are sized for one max_tok
  piece, not an8K span. `BatchBuilder` derives seqused from the rebased first position and reuses
  per-set upload/mirror state (`batch.cpp:812` forseqused). Span preparation must not free/rebase
  an earlier saved view.
- `Scheduler::reserve_ckpt/plan_chunk`, `checkpoint_copy`, token history and EORS callbacks are
  per completed ordinary chunk. They cannot be called midspan or replayed twice. A generic
  quiescence hook alone does not provide a span transaction or rollback authority.

## Ordered realization and gates

The metadata helper is the independent source stage now ready. Next, allow about one focused
source day for the read-only diagnostic plus host malformed/stale-frame and actual routing-witness
cases, then one bounded logging off/on window. This establishes whether repeated expert bytes
justify the executor. No favorable union/copy number is claimed in advance.

If justified, a realistic serial engineering sequence is1–2 days for span batch/KV ownership and
budget/abort host cases,2–3 days for the native optional span entry and owned captured frames,
then1–2 days for the fresh8K producer-only integration and independent per-piece/card/model gates.
These are effort estimates, not promises; failed lifetime/cancellation proof blocks enable.
Resumed prefix/EORS/checkpoint transactions and a smaller rolling window are later stages,
roughly several additional focused days, not prerequisites silently assumed solved.

Require original piece outputs/routing/sorted/gather bytes, logical window/compressed/index KV,
rawF32 tail/token state and logits equal the proven4K reference, including zero/dead/padding rows.
Then longKLD/Kh, fresh6817/8K,96GiB actual miss/packing exceptions, cancellation between pieces/
waves and allocation refusal. A future prefix-capable stage additionally needs cold versus exact
resumed/EORS/grid16 and checkpoint-failure cases. Price complete TTFT including all metadata,
snapshot/restore and residency costs; run N4/N8 with flag0 and the enabled-but-ineligible scope
to expose permanent memory costs. Keep original4K as fallback and leave every active switch off
until its specific source/build/card/model cohort passes. The pure helper is not such a gate.
