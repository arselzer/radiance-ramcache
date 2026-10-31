# DS41 packed native producer-output experiment

Source only; no performance or byte-identity claim before the gates below pass.
`RADIANCE_EXPERIMENTAL_DS41_KV_PACKED=1` is strictly opt-in; unset/`0` preserves the
original graph, buffers and issue order. Any other value refuses startup. The first
implementation admits width512 compressed keys (format1, E4M3 scale/group16, 288B/page)
and width128 index keys (format2, original power-of-two scale/group32, 68B/page),
native ratios1/2, one entry per page. It imposes no RAM/cache-capacity constant.

The logical dtype stays BF16. Original FP4 codes/scales are captured beside the
unchanged quantizer's BF16 output in transient work-row buffers. The store checks
reconstruction against the actual old entry-store BF16 bits, then writes a
self-contained page. It never requantizes BF16 to infer codes. Zero pages decode
to +0. Signed zero, minimum-scale FTZ and exceptional index Inf-scale products
use the shared codec's exact representations; captured literal pairs preserve
actual NaN payloads. `PackResult::Raw` is an invariant failure for the admitted
closed producer domain, **not a supported mid-request fallback**. A nonzero Raw
count or any bit mismatch in the independent oracles blocks model testing.

Window keys keep the existing FP8 roundtrip/BF16 cache. Compressor rings/tails
stay F32. A disabled packed flag is the BF16 fallback. No permanent BF16 history
shadow is allocated. Core physical-page ownership, copies/COW/prefix snapshots,
hashes and fail-closed issue guards come from the matching page-plumbing cohort.
Packed streamed KV is refused for this first experiment.

All original reader dispatch choices remain: scalar index/attention, fast decode
and WMMA. Each cache load reconstructs BF16 bits before the original FMA,
reduction or matrix-fragment operation. In WMMA attention, the packed path
emulates the old 8x8 BF16 transpose using integer pointer/format shuffles; mixed
window/packed tiles are covered. No FP4 arithmetic replaces BF16 arithmetic.

New required op names: `quant_roundtrip_kvpack`, `entry_store_kvpack`,
`index_score_kvpack`, `attn_sparse_sink_kvpack`. Optional fused op names:
`rmsnorm_rope_roundtrip_kvpack`, `rope_rows_roundtrip_kvpack`; refusal falls back
to the original norm/rope chain followed by the captured roundtrip. A missing
required packed op refuses startup. The per-rank startup line starts
`radiance: deepseek41: KV_PACKED rank=` and records format IDs, row bytes and
group counts. A successfully enqueued normal store/index/attention call prints
once per rank and role: `[r4d] ds41 KV_PACKED active rank=... role=...` (stores
also once per format). These markers prove normal dispatch was queued; byte
oracles and completed model requests prove execution. Disabled runs print none.

Mandatory validation, all serialized under AGENT-LOAD-RULES; every GPU command
inside the stopped/restored-QFN two-lock window:

1. Integrate codec + page ABI + producer/reader cohort and rebuild every plugin
   together. Host `ds41_kv_codec_test`, page/issue/planner fixtures and
   `ds41_work_rows_arch_test` must pass; keep `-LE gpu` on host ctest commands.
2. Build `kv-packed-paths-check` through `work/build.lock` into a build directory.
   Run `ds41_kv_packed_paths_check LIBR4D.so` in separate processes with pins
   unset, with `EXACT_TALL=1`, and with adopted `LCP=2048`/`N128=1` (full
   `RADIANCE_EXPERIMENTAL_DS41_` env prefixes). This composes the codec's full
   code/scale-domain and all65536 BF16-input producer/store gate with fused rope
   capture, ordinary/forced old+N128 norm capture, scalar/WMMA index and every
   reachable scalar/fast-decode/WMMA attention variant. All comparisons use
   uint16/byte equality, including NaN payloads, pitched rows, skipped/zero
   pages, guards, ratios1/2 and mixed window/packed transpose boundaries.
   Checker exports have no registered schema/row and cannot be selected by the
   architecture. Unsupported checker calls fail the gate; no silent fallback.
3. Only after those pass: one plain/packed model window on the current192GiB
   configuration: long KLD and harness Kh/replies/counts, prefix resume/COW,
   DSpark off/on, N4/N8; confirm startup and actual-path markers on every
   attending rank, all same-library cohort, lower physical KV allocation and
   no reply/token/logit change. Check a smaller startup cache/miss path too.

This branch intentionally contains no harness arm, chosen default, long-context
speed forecast or change to query/verification/prefix policy.
