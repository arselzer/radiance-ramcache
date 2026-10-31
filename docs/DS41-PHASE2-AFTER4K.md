# Phase2 after the first4K window

Source/design only. No project code was built, syntax-checked or executed; no tests, simulations,
emulators, model requests, GPU actions or services ran in this lane. A bounded read-only parse
of existing small chain38 logs/receipts and direct integer budget arithmetic supplied the numbers
below. Running cohort0f62a578 is distinct from audited source0d27ba75. Full window validation and
adoption belong to root; these timings alone are not a passed prefix/spill/reply gate.

## Decision from current evidence

The original500tok/s long-prefill target is already exceeded by the first4K pair on current
256GB/192GiB settings:3086 tokens /5.690759s TTFT =542.3tok/s,6817 /12.687156s =537.3tok/s.
The matching2K times are7.325505s and17.120403s. Phase2 is therefore further long-span/traffic or
memory work, rather than a prerequisite for the500 target. It must earn its executor complexity
and any persistent resident displacement against concurrent decode, which remains first priority.
This pair is not a uniform short-prompt win:82-token TTFT1.912→3.477s,933-token4.255→4.368s
and1890-token4.974→4.999s. The first short sample includes fresh-server effects; no causal
attribution or stable regression percentage is established by that one pair.

A remaining mechanism is real: the6817-token4K run uses2736 +4080 card rows, so producer layers
0..19 run twice before the consumer suffix. All experts are already on the cards for those tall
passes; simply re-enabling a card path, RAM cache, read-ahead, bigger ring, or a cache-policy retry
would not eliminate those second transfers. Phase2 would retain activations and reuse a loaded
expert across the original pieces while executing layer-major over a bounded span.

## Actual chain38 traffic

Artifacts: `tools/ds41-host-experts/build/ab-20261031T162629Z` in the integration repo. The paired
names contain `m196608`, `Lc2048`/`Lc4096`, `P1-Qe64-Mb1-Cs-Kh-Lm8192-Ulong8k`;4K also has
`Pci4096`. They keep Rb/Vp/N128 off. The following are complete shutdown counters for that single
6817-token request, not totals from the five-document Ulong set. GB is decimal and log-rounded.

| Rank | P1 card layers2K→4K | RAM units2K→4K | Demand drive units2K→4K | All drive reads2K→4K | Ring copies2K→4K | DMA GB2K→4K | Waves2K→4K |
|---|---|---|---|---|---|---|---|
|0|98→57|4228→2761|1137→757|1470→927|5370→3522|96.67→63.40|2723→1786|
|1|98→57|4309→2806|1147→762|1464→915|5464→3574|98.37→64.34|2770→1816|
|2|98→57|4300→2764|1129→758|1493→943|5435→3525|97.84→63.46|2753→1793|
|3|98→57|4303→2767|1147→761|1504→943|5455→3533|98.20→63.60|2771→1795|

Four-rank totals: ring copies21724→14154 (−34.85%); DMA391.08→254.80GB (−136.28GB,−34.85%);
waves11017→7190 (−34.74%); actual drive reads5931→3728 (−37.14%). Demand units4560→3038
are CPU-packed, with0 block fallbacks in both arms. Host-pool units24→18 complete the copy
counts. Actual reads exceed demand by dropped read-ahead units:1371→690. These counters do not
identify expert unions or an exact number of duplicate expert IDs, so254.8GB cannot all be
called redundant traffic. No gain for eliminating the remaining repeats is measured here.
There is no independent P1 relay-byte counter; packed-unit and block-fallback counts are the
available relayout evidence, and should not be renamed as a fabricated relay measurement.

Per-rank4K chunk wall spans are12.53..12.60s, drive busy63.9..76.3%, issuing-thread drive wait
2.189..3.069s, packer3.40..3.67ms per demand unit, and ready-slot wait1.907..3.475s. The2K
spans are16.96..17.03s, drive busy79.9..87.1%, drive wait4.461..6.343s and ready wait1.077..
1.672s. These overlap; summing them would invent a bottleneck decomposition. Smaller drive
fraction and larger ready wait mean a constant-bandwidth linear speed forecast is inappropriate.
The4K ring remains12 physical slots/waves2; its64-slot lent arena is a decode return mechanism,
not a64-slot prefill supply ring. The log says64 fit at4K while2K has43..44 fitted arena slots;
Qe's cap itself was unchanged. Do not attribute DMA reduction to an enlarged prefill ring.

The five-document Ulong192GiB aggregate falls from132→111 card layers and about129.0..129.7→
115.0..115.5GB/card, with only the3086-token document losing a chunk. At96GiB the same Ulong
copy counts/bytes remain unchanged, but demand drive units rise to2987/2966/2937/2948 per rank
at4K, versus3250/3305/3406/3516 at2K. Long3086 TTFT22.058749→14.525250s. This is concrete
proof that a Phase2 planner must preserve configurable cache budgets and the original miss path;
it cannot assume the192GiB source mix or call all units RAM hits. Root owns the functional gate.

## Closed retained-byte inventory

Actual source seams are `deepseek41.cpp::buffers`, its `run_layer` Seg1/Seg2/Seg3 and fused
next-layer post/collapse/norm, plus `MoeFP8::pass_head/pass_tail` in `rad_block_moe_fp8.h`.
A conservative retention cut keeps the following across original pieces:

| Per row | Bytes at D5120/hc4/top6 |
|---|---:|
|h0 +h1 streams, BF16|81920|
|next layer's xn +current FFN input, BF16|20480|
|six sorted-slot expert down outputs, BF16|61440|
|shared output +shared unity gate, BF16|10242|
|pre_a/pre_f/post/comb, F32|112|
|routing IDs/F32 weights/sorted slot indices|72|
|index_sel,512 I32 entries|2048|
|index_keep at max_ctx8192,candidate ratio1/block8|132|
|**Total**|**176446 =172.3105KiB**|

The previous172.3KiB estimate is a sound rounded conservative inventory at8K context, unlike the
old80/122KiB proposals; explicitly retaining shared_one adds2B/row. Candidate keep is variable:
`ceil(ceil((max_ctx/ratio+1)/block)/32)` I32 words. At32768 it is129 words; at1M it is4097.
A constant172.3KiB estimate at all contexts would underfund this live cross-layer state.

The current native path has no inverse-sort buffer and no separate FP32 shared combine. Current
router arguments write/read only local96 counts and97 offsets; retain those live prefixes per
ORIGINAL piece (772B/piece), independently from each piece's sorted array. The helper does not
create a segment-wide sort. Different geometry, representations or combine forms need their
own inventory, rather than silently reusing this one.

GU, SwiGLU intermediate, shared-up, router logits, q/kv/attention, score plane and compressor
work scratch can stay piece-local after their consumers finish. xcol is not a retained residual
in the native external-input/no-add MoE. The next-xn output cannot simply be dropped: the fused
post/collapse/norm creates it in the preceding layer. All rawF32 compressor/tok state remains KV
state with its old rounding/order. Final-hidden/logit rows, piece positions/tokens/cuseqlens,
KV tables, copy/capture metadata and checkpoint staging are additional explicit costs, not free
space hidden in the per-row estimate.

`core/place/ds41_phase2_retention.h` is a NEW unused pure prerequisite. It computes aligned slices
for the conservative inventory, per-piece local route metadata, overflow-safe totals, and an
explicit device/host funding check. It has NO allocation, environment flag, kernel, Engine,
Scheduler, arch or P1 caller. Host funding preserves future expert-cache growth plus configured
system headroom. Arena replacement bytes default0: no sharing/union is inferred from a modelled
budget. Prepared source cases cover field presence, variable context, alignment/non-overlap,
shape/span/overflow refusal and independently variable future-cache/headroom bounds. No case ran.

##8K versus16K bounded spans

Raw retained state, before per-buffer alignment and route metadata:

| Rows | Max context | Retained MiB/card |
|---|---:|---:|
|3086|8192|519.287|
|4096|8192|689.242|
|6817|8192|1147.110|
|8192|8192|1378.484|
|16384|16384|2758.969|

First semantic prototype should preserve the ORIGINAL <=4096 piece boundaries/heights and run
at most8192 rows of one sequence, no drafting/media/LoRA/concurrent decode within that span.
It costs another689MiB of retained planes over the4K subset, plus any larger window working set
and frame metadata. Plain WINDOW capacity can add about160MiB/card when its declared live-step
range rises4K→8K, and another480MiB for4K→16K (40 layers×1024B/row; exact planned tables/padding
must supply actual costs). Roughly850MiB additional at8K or2.49GiB at16K therefore needs an explicit
resident/VRAM plan; no unused card memory was measured. Existing Qe64 cannot automatically turn
all added space into decode residents. Extra permanent reservations require N4/N8 validation.

A later256-row dense-piece variant would model arena157.24MiB plus replacing its43.08MiB retained
subset by1378.48MiB, about1492.65MiB total arena at8K. A16K plan must add its2758.97MiB retained
planes to an independently planned context-dependent piece arena. These are storage models,
not current allocations or byte gates. They exclude ring/KV/final-hidden/frame/checkpoint costs
and need separate height/dispatch equivalence. The old0.95GiB8K total is not a usable budget.
8K is the smallest useful next span for the6817 prompt;16K should wait until8K supplies unique-ID
traffic evidence, actual allocation and an E2E win. Retaining some planes in pinned host memory
could reduce VRAM, but incurs copies and host headroom costs; budget refusal must fall back to
current4K without trimming a user's expert cache or dropping the128GB/96GiB spill path.

## Exact execution contract still to close

1. Retain original dense/norm/rope/compressor call heights, absolute positions, sequences, causal
   bounds, key/full consumer suffixes and FP32 tail order. A single16K MoE re-sort is unnecessary
   and would exceed current sorted-height gates. Replay each loaded expert on each piece's
   ORIGINAL stable sort and six-slot buffers, with the old SwiGLU/down/gather and ascending-rank
   reduction order. Release a ring slot only after every piece consuming it has finished.
2. Split route/shared preparation from expert replay and gather. Existing Seg1/Seg2/Seg3 is a
   useful seam, but `MoeFP8::pass_head` still combines router, expert launches and shared/mix.
   P1 currently retains ONE captured launch triple per layer/pass; multi-piece frames need owned
   tensor/param arrays, local counts/sorted offsets and independent live pointers/events. No
   borrowed stack RadArgs or overwritten Ctx route-count row may survive into another piece.
3. Reserve/adopt/hold every piece's real KV IDs before writes. A later piece cannot free/reuse a
   WINDOW block while an earlier piece still needs its other layer stripe. Current per-step
   Scheduler ensure/slot mappings, compressor work arrays and state indices are not a span API.
   Conservatively retaining original full-span WINDOW capacity is simpler than claiming a tiny
   rolling window before its lifetime proof exists. RawF32 tails and unwritten padding stay exact.
4. Engram IDs can be prepared in original token order because they depend on tokens/history, not
   activations. However that advances per-layer token history; intermediate checkpoint/cancel
   rollback must not see future history prematurely. Preserve the existing Engram split and
   rank/column layout; whole-segment dedup was proposed, not measured, and is not a prerequisite.
5. Prefix/checkpoint publication requires all layers of a logical chunk to have completed. A
   layer-major span has intermediate per-layer state, not a publishable checkpoint. Stage each
   required raw state/WINDOW snapshot with ownership, publish atomically only when complete,
   or explicitly refuse the experimental mode where that cannot be done. EORS/grid16, resume
   suffixes, cancel/reap/COW refs and partial failure need a complete transaction design first.

These gaps block active execution, not the pure accounting helper. No runtime enable is added.

## Prior attempts and decisive next proof

The full existing PREFILL-200-DESIGN (including critique), PHASE1-PLAN and NEXT-STEPS were read.
Prior archive audit covered all31reports and full NIGHT-LOG. Main distinctions remain: Phase2
has been proposed but not implemented/measured; current LCP/P1/end alignment, last128 admission,
live-tail/balance, packed RAM, asynchronous sources, causal suffixes and arena lending already
exist. Whole-unit read-ahead, alternative cache policies, TBO, ring/grace retries and extra cache
capacity were measured/replayed without useful benefit; no such retry is justified here.
Old128GB/96GiB drive floors and optimistic256GB forecasts are not current measurements. Early
P0 watchdog-contaminated transfer/kernel figures remain caveated, not fresh independent evidence.

First prototype gate, after state contracts close: isolated8K span, original4K pieces, default
OFF. Per original piece/layer record expert IDs, source form and actual copied bytes so the
union saving is measured rather than guessed from aggregates. Require original piece outputs,
local sorted/gather bytes, every raw state/WINDOW/compressed/index KV byte and logit record equal
4K plain/P1; then longKLD, Kh, resumed prefix/EORS/cancel and96GiB miss cases in one guarded window.
Price complete TTFT including checkpoint staging/copies, plus N4/N8 decode and resident restoration.
Stop on any byte difference or if saved DMA does not pay for extra state traffic/metadata work.
The retention helper alone grants no authority to call an executor, allocate unbudgeted space,
change recommended configuration or claim a Phase2 speedup.
