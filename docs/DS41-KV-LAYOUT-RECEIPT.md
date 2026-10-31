# Exact growing-KV layout and Kh capture receipt

Source only, off44f50fb4; no build, syntax check, test or card/model invocation. No ABI structure
or model/kernel arithmetic changes. `RADIANCE_EXPERIMENTAL_DS41_KV_GROW_LAYOUT=1` is a separate,
default-off audit switch, required for both cold/default-backed and live runner roles. Absent/0
emits no new startup/Kh rows and performs no new copies, drains or allocations. Invalid values
refuse the requested audit rather than silently enabling it. Current diagnostic scope is native
DS41 TP4, deterministic, max sequences1, speculative0, actual elastic card VMem, no zero/streamed
groups. Metadata bounds64groups/128layers/2048buffers/256-byte names keep output finite.

`DS41_KV_LAYOUT` reports each rank's actual Pool logical budget, used carve, backend-rounded VA,
actual backend granule/physical committed bytes and base, max context/step/sequences, planned
device/fixed/host arena and scratch bytes, current free/total VRAM, deferred step/lazy budget,
prefix/checkpoint geometry and local logits width. Group rows report exact layer IDs, physical
page/state counts/bytes, offsets and layer stride. Buffer rows report actual buffer-plan byte
sizes/offsets/domain/kind, with names encoded as hex (no ambiguous whitespace). Buffer lifetimes
can overlap in the arena: these rows are not an additive process-memory figure.

The exact cold reserved VA is the only authority for a paired live MAX_MIB. humanb() rounds to
two-decimal GiB at large context; /stats carved bytes omit reservation slack, so neither can
authorize MAX. The runner must require all four exact cold reservations/granules and layouts
agree, freeze the whole reservation explicitly, and reject a live layout change. All extent,
stride, snapshot multiplication and bitmap-capacity arithmetic is guarded in64bits. The checked
helpers have3 prepared host cases in `kv_layout_receipt_test` (flag parser, overflow/nonmutation,
logical versus physical tail extents); they have not run.

`DS41_KV_SNAPSHOT` reports actual opened slot count/width/payload and actual HostArena bitmap
vector capacity, object bytes and metadata allocation payload. Allocator private headers,
PrefixCache maps and opaque driver capture/code allocations are explicitly untracked. Runtime
process RSS/VmLck/MemAvailable and free VRAM samples are still required; these diagnostics do
not pretend to give an exact whole-process footprint.

Kh's fetch keeps a pinned host buffer per rank as large as the largest KV layer stripe or
local logits row. This remains true in the sparse live path, even though only referenced FULL
runs are copied. The layout gives that exact future capacity from the actual plan; approximately
4GiB total can be pinned at BF161M context. It is additional debug memory, not reduced by sparse
copying. Root must fund it in host/driver headroom before requests. Cold Kh copies whole layer
stripes; live FULL copies only referenced physical runs. Their timing difference belongs in
the full lifecycle receipt and is not a serving-speed claim.

Under the audit switch, every attempted Kh fetch logs `DS41_KV_HASH_COPY` with actual rank,
sequence, group/layer (group0/layer-1 is logits), requested bytes, allocated capacity, successful
copied bytes, sparse/run count and ok status. A failed allocation or copy is explicitly ok0;
its byte count is not a successful capture. The runner requires exactly every expected group
layer and local logits row from the actual layout, all successful. This closes a previous
diagnostic gap: two failed huge captures must not pass merely because they folded the same
zero hashes. No additional copy or numerical operation is introduced by the receipt.

The same audit flag adds `DS41_KV_CANCEL` only after the scheduler actually closes a previously
Waiting/Running/Preempted request as Cancelled, under its ownership lock. It records the exact
sequence, step, prior state and deferred-cleanup flag. Finished, already cancelled, failed and
not-found requests emit no accepted receipt. A stream-disconnect model gate requires prior
Running (state1), active/accepted/cancelled1 for its observed sequence, a strict partial text
prefix and remaining generation budget; the HTTP disconnect counter alone is insufficient
because it also increments for a late close whose sinks already completed. No new endpoint,
state transition or numerical work is added. Flag-off cancellation emits its original log lines.

The first large-capacity matrix keeps M16, F32 scores and all independent memory switches off:
INDEX_SCORE_COMPACT0, INDEX_SCORE_BF160, packed main/index/window0 and packed-fast0. Actual
`index_plane_rows=min(max_tok,max(legacy256,256MiB/4/columns))` gives irows16/one run at M16.
At512Ki/1Mi context, F32 score storage is33554496/67108928B; native candidate-block8 masks are
131136/262208B; two64-column F32 RoPE planes are256/512MiB. The exact buffer receipt must agree.
These are source sizes, not a filled-context serving proof. Full BF16 history remains3200B per
input token/card, with actual page alignment/fixed window/raw state/rounding additional.

Do not enable compact rows or B16 to rescue a refused fit. Compact64 needs independent LCP
dispatch/KLD/Kh proof; B16 retains row geometry and initially refuses packed writers. Packed
window/main/index must pass their independent current32K byte gates before any large-capacity
combination. The first growing runner is plain BF16 and tests short original prompts at a
configured512K/1M maximum, never claims filled512K/1M context correctness or performance.
