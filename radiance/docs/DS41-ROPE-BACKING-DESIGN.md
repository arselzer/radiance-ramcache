# DS41 RoPE backing feasibility, 31 October 2026

Status: source feasibility and design only, based on `0f62a578`. No runtime
patch, enable, build, test, emulator, GPU operation or measurement. Every reply
must remain byte-identical to the merged base. Current 256GB/192GiB throughput
is the priority; configurable smaller caches remain supported.

## What exists and what was tried

`HANDOFF.md`, `NEXT-STEPS.md`, `LONG-CONTEXT.md`, the archive's RoPE references,
the relevant NIGHT rows and local upstream/current source were reconciled.
Existing exact norm/RoPE/quantization fusion, n128 work, consumer cuts,
compressor work-row sizing and resume-grid changes are not missing features.
The archived `0da35f7d` change documented table memory and resumable positions;
it did not change physical backing. The qsa work-row experiment recorded at
NIGHT approximately02:45 Oct31 compared the resulting angle table bytes when
reducing padded work, not a sparse table allocation. No sparse-RoPE engine
trial was found in these sources. Individual raw window captures and all
historical patches were not exhaustively re-read.

Local `upstream/main:core/runtime/arena.cpp` and the current implementation
both allocate the complete ordinary arena once. The source already generates
angles lazily. `r4d_rad_rope_table` launches over only its supplied positions;
`r4d_qk_rope_table_kernel` writes each valid absolute position at its own table
row, and skips negative/out-of-height positions. Other rows keep their bytes.
Adding another on-demand trig generator would duplicate existing behavior.

The two DS41 persistent buffers are declared by `buffers()` in
`arch/deepseek41/deepseek41.cpp` as `rope_window` and `rope_compressed`, each
`[max_ctx, rot]` F32, with native rot64. Their combined logical payload is
512B/position/card:16MiB at32Ki,256MiB at512Ki and512MiB at1Mi. Neither table is
divided by TP4. These are source sizes, not measured process allocations.

## Arithmetic and access contract

Reuse the original `rope_table` operators and readers unchanged. Preserve
absolute positions, theta, denominator, float casts/division, `powf`, YaRN
frequency and magnitude operations, `sincosf` and stores. No CPU/GPU trig
substitution or table rebasing is justified by this audit. The existing
`ds41_rope_rows`, fused norm/RoPE/roundtrip and inverse output-RoPE paths all
read the table using absolute positions and full pitches. Cached K values are
already rotated; attention does not require retaining old RoPE table rows to
read historical keys.

The complete declaration/issue families are:

- Trunk `m.rope_a` and `m.rope_b_step` write the current batch positions before
  block work. Query, window key, index query and output inverse-RoPE consume
  the corresponding table and row positions.
- Each compressor source's `L.c_tab` rewrites compressed-table rows at its
  `bpos`, after the earlier query/window reads. Latent and index-key RoPE then
  consume those rows. Preserve that order.
- DSpark `d.c_tab` and `d.q_tab` write the shared window table for their
  context/query passes. Their readers use that pass's positions. Deferred
  context can revisit an older range; query/issued-ahead positions may be
  computed by device advance rather than known exactly by the host.

`qsa_work_kernel` in `libr4d/r4d_qsa_work_bf16.hip` writes a completed block's
first-token position `b*ratio`. It can precede the first token of this batch by
up to ratio−1. Padded work writes `bpos=0`, even when page=−1; a following
`rope_table` can therefore write row0. Every backing policy must include row0,
not infer table access solely from live KV pages or current query positions.
Native DS41 passes no multi-axis rope operand to this work operator.

## A safe first bound without per-token synchronization

Restrict a first version to native no-drafter/no-speculation, no tree/TBO or
media-position remapping. For ordinary host-built rows, BatchBuilder stages
`h_pos=e.ctx_len+j` for `j<e.n_tokens`. Its `max_ctx_len` is an upward bucket
over all entry context lengths, and `max_q_len` bounds every entry's token
count. Thus every valid position satisfies

`0 <= p < max_ctx_len + max_q_len`.

Issued-ahead no-spec decode also has a conservative host ceiling:
`Scheduler::step_next` sets `ctx_up=r.n_computed+e.n_tokens`, explicitly the
furthest context the preceding step can commit, and uses it as the next
entry's `ctx_len`. Device `ADV_TRUNK` can choose an earlier real position;
it cannot exceed that ceiling under this scope. Prefix resume and mixed
prefill/decode retain the ordinary entry bound. Completed compressor first
positions never exceed the batch's last position; their older backoff and
padded row0 are covered by a prefix.

The initial target is therefore both tables' prefix
`[0,min(max_ctx,max_ctx_len+max_q_len))`, using checked integer arithmetic.
Keep a host high-water mark and invoke a mapping operation only when that
prefix reaches a previously unbacked physical granule. No lane drain,
position readback or other GPU operation is needed on a steady token.
The same pre-pass check must run before captured replay, not only while a
pass is recorded. Unknown batch/frame construction must be refused rather
than trusting unproved bounds. Draft/context/advance bounds need a separate
proof before extending this scope.

Draining both lanes and reading device metadata on EVERY token could establish
actual sparse row sets, but introduces a host round trip into a path built to
avoid one. This is not proposed as a current throughput optimization for the
16MiB table cost at32K.

## Allocation and lifetime design

A possible implementation keeps the existing arena VA, buffer offsets, table
views and full logical shapes. Reserve its complete maximum range through
VMem, then immediately back every non-table and boundary/shared granule.
Only granules wholly contained within one admitted persistent RoPE buffer may
be left unbacked. With buffer interval `[o,o+n)` and device granularity G,
the eligible interior is `[ceil(o/G)*G,floor((o+n)/G)*G)`. An empty interior
has no saving. Other persistent buffers, arena scratch, transients and P1
lending regions remain backed as before. Never decommit a granule containing
any of their bytes.

Monotonic prefix commits preserve all previously accessible pages and stable
VA while earlier work is still in flight. All rank commits must succeed
before any new cohort pass can read that prefix. Partial commit/map failure
is fatal; it cannot become a smaller usable table or an eviction retry.
No mapping change can silently alter kernel parameters or truncate a tensor.

Retain backing through the request. A later shrink may run only at an actual
all-stream/all-rank quiescent idle boundary, accounting for deferred/context,
captured, mover and inspector access. The existing VMem failed-unmap ledger
fix `29319f85` is a prerequisite for that phase. A first monotonic version
need not shrink until shutdown. A stable address reservation does not itself
prove that decommitting current contents is safe.

## Budget and admission gaps

1. Current `Arena::init` uses `rad_dev_alloc` for the full range; `Ctx::bind_arena`
   and `arena_plan_bytes` share that exact full-allocation size. A VMem variant
   must charge the rounded full reserve/maximum backing explicitly, including
   final granule slack. Preserve the default allocation path byte-for-byte.
2. Keeping this maximum startup budget protects eventual fit but gives no
   additional planned expert residents. Actual unused backing can fall, yet
   the expert pool remains unchanged. A monotonic prefix reaches the full
   512MiB at1M; it does not reduce full-context peak table memory. Reusing that
   budget needs a separately proved rolling live-range bound and coordinated
   recall/funding before any new table page is mapped.
3. The final Program does NOT retain complete per-op buffer reads/writes.
   `Builder::op_bufs_` contains them during declaration, then only liveness
   endpoints survive in BufferInfo. Consequently table names and known RoPE
   op names cannot prove that no new/unknown op reads an unbacked table row.
   Add a narrow Builder-produced private inventory, then admit every real
   reader/writer and selected band, with no unknown/tuned selection and actual
   compatible module provenance. A name-only whitelist is insufficient.
4. Full-buffer debug copies and per-op oracle/repeat can read unbacked rows.
   Refuse them in the first version or provide explicit backed-range handling.
   Include buffer dump paths, not only the normal model's numerical kernels.
5. Every allocation/host metadata amount must be bounded and charged. The
   design must preserve smaller expert/KV budgets and avoid any256GB/192GiB
   assumption. An unsupported VMem device must refuse explicit activation or
   report whole-backing fallback honestly; an enabled flag proves no saving.

Because the reader inventory is missing and maximum-budget prefix backing
does not create resident capacity, this lane stops at the durable design.
There is no unsafe sparse Arena patch or recommended runtime switch. Proposed
name `RADIANCE_EXPERIMENTAL_DS41_ROPE_PAGED` is reserved only in this document;
it is not implemented.

## Required source and validation work before enable

Host cases should prove complete-use admission/refusal, no-spec ordinary and
issued-ahead ceilings, mixed/prefix resume, compressor backoff/padded row0,
checked arithmetic and guard-column/offset geometry. Exercise32K/512K/1M,
arbitrary device granules, zero eligible interiors, boundaries shared with
other buffers, monotonic map counts and partial-map fatal propagation.
Off/no-Vmem/unknown-reader/oracle/TBO/draft controls must stay explicit.

A strict card oracle must compare original full-backed generation against
the same generator over sparse backing, byte-for-byte in every produced F32
cell. Include ordinary/compressed theta, current YaRN settings, zero and
padding positions, boundaries around65,536 and1,048,576, granule crossings,
reversed position order and captured replay with changing payloads. Out-of-
range guards and unrelated arena bytes must remain intact. Any later idle
shrink needs delayed-reader and unmap/map fault cases before a model gate.

Only then run independent long KLD and exact replies/KV hashes across cold,
prefix/resume, mixed N4/N8 and reduced-cache miss arms in a normal two-lock
QFN stop/restore window. Compare actual startup/peak backing and resident
counts separately from maximum reservation. No runtime or speed estimate is
established by this design.
