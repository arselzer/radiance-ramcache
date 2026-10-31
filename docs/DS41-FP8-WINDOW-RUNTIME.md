# Native FP8 WINDOW storage candidate

This is default-off source, not an adopted optimization. No build, host test, card oracle,
model run or performance measurement was executed while preparing it. The independent codec
source is `98f18434`; its numerical domain is still unproved on the cards. In particular, the
sign/payload of a NaN after the original E4M3 decode and GPU float multiply has no source/ISA
guarantee. A negative code-NaN in a group with several finite outputs could fail the candidate
normal decoder and exceed the two-pattern escape. A nonzero Raw count blocks activation. Do
not alter original FP8 bytes to hide that failure, tolerate numeric equality, or retry a partly
modified sequence. A different captured exception format would be a separately reviewed change.

`RADIANCE_EXPERIMENTAL_DS41_WINDOW_PACKED=1` requires `KV_PACKED=1`, the native 512-wide
main window, and the coherent new core/arch/libr4d cohort. Missing, `0`, and the absent arm part
preserve plain window storage. Other values, incompatible plugins, unknown/tuned/fallback rows,
wrong geometry, and incomplete per-layer producer/store/reader chains refuse before allocation.
The first harness profile is TP4 with all four attention ranks active; attention-zero layouts
are not covered by the model gate. Cache budgets remain variable; no RAM/cache capacity is fixed
by this feature. `KV_PACKED_FAST` still selects only the independent FP4 decoder. FP8 windows use
the reference integer decoder under both settings and receive separate combined card gates.

## Physical and numerical contract

ABI storage format3 (`RAD_KV_STORAGE_DS41_FP8_WINDOW_G32`) applies only to logical BF16
`RAD_KV_WINDOW`, one KV head, logical head_dim256, the original sixteen slots per page. A native
512-wide row stores original FP8 E4M3 bytes512 plus UE exponent bytes16, or528 bytes. A physical
layer page is8448 bytes, compared with16384 bytes before (48.4375% fewer payload bytes). Pool,
fixed-window floor, layer stripe, checkpoint and transfer costs derive from this physical pitch.
The index window, DSpark window and raw FP32 compressor tails retain their original formats.

The same original FP8 scale/code expressions run after each unchanged producer BF16 expression.
The pass-local capture has `[work_rows,528]` bytes and the same row offset as its actual BF16
source; the store consumes it before reuse at the next layer. There is no permanent BF16 cache
shadow. `kv_store_windowpack` reads the full512 actual uint16 values and original capture, then
uses the same slot mapping as `kv_store`. It preserves raw uint16 bits: reusing `entry_store` would
be wrong because that path performs another BF16 narrowing/NaN quieting. Negative/out-of-range
slots store nothing, exactly as old `kv_store`; untouched zero pages reconstruct +0.

Normal descriptor bytes105..247 reconstruct by integer operations. Source255 (+Inf) can use
scale0's two actual-bit patterns and32 selectors; scale0 of a zero page is +0. Invalid descriptors
trap in the device reader rather than silently returning zero. Every actual output is checked
against reconstruction before storing. Raw is an invariant violation after a proven closed-domain
gate; it is not an exception fallback, an abort/retry mechanism or permission to lose bits.

The original attention source and argument layout remain unchanged. A separate new include/namespace
copies its scalar/WMMA arithmetic and dispatcher, so ordinary attention acquires no window-format
branch. Scalar, decode/vector, transposed-load and WMMA paths reconstruct BF16 at their original key load
seams. Existing dispatch thresholds, scalar FMA/reduction order, online-softmax code and WMMA
fragments are retained. Packed transposed loads use the already introduced integer8×8 shuffle
mapping, with per-source window/compressed format selection even in mixed tiles. The old plain
window/FP4 reader serves as the independent reference in each forced form.

## Ownership, copies and admission

`KVManager` still owns fixed window pages. COW, prefix window checkpoints and tier fragments copy
whole self-contained physical pages; no side pool, relocated pointers or extra mutable scale
state exists. `checkpoint_copy` uses `layer_bytes_per_block()`/`bytes_per_block`, including each
layer stripe. Prefix restore thus restores the exact code/scale/literal bytes. Logical debug
hashes decode each of the sixteen rows into one temporary page and preserve the original hash run
boundaries; they do not hash the physical encoding. Growth holder inventory keeps the complete
non-FULL WINDOW pool/floor mapped using its derived physical bytes; ordinary FULL-page growth
does not unmap live window bytes. Streaming/replicated tier paths retain their packed-format
refusals; no new streaming composition is claimed.

Before plan/allocation, `ds41_window_packed_admit` proves the native declaration and every selected
band's exact op/row/export, untuned device-only implementation and one actual loaded libr4d DSO.
Every actual RAD_KV binding independently requires its dedicated op/row/operand and explicit
declared/resolved format: store cache3 uses `kv_format=3`; attention window1 uses `window_format=3`,
while compressed cache4 separately requires `kv_format=1`. A plain fallback never sees a shortened
allocation. The first native runtime requires the actual fused norm/rope producer. An absent or unresolved
fused alias refuses; it cannot inherit an unproved norm/rope fallback from another library.
Standalone quant/rope capture exports remain independently oracle-covered for later composition.

## Source verification and required serialized queue

Rebuild the entire core, architecture, libref and libr4d cohort following `work/AGENT-LOAD-RULES.md`.
Host targets are `ds41_window_packed_test` (three cases), `ds41_kv_page_test` (one new native-window
physical/checkpoint case), existing `kv_storage_guard_test`, and `ab_window_window_kv_test`. Run
only exact host test names with `-LE gpu` under `build.lock`; these sources have not been executed.
The recording/shape probe includes one new cut/row-offset/sizing case in
`ds41_shape_probe_arch_test`; it and the existing FP4 packed/fast suites must also continue to pass.

Card sources/targets under `tools/ds41-host-experts`:

- `window-codec-check` / `ds41_window_codec_device_check LIBR4D.so`: independent original quantizer
  transcription/domain, all65536 BF16 input patterns across four4096-row profiles, literal-pair
  cases, guards, strict zeroRaw.
- `window-packed-paths-check` / `ds41_window_packed_paths_check LIBR4D.so`: first runs that independent
  gate again, then84 actual old/new producer+store cases (standalone quant, fused norm/rope, fused
  rope; heights1,7,33,49,257,2048,4096; four input profiles). Full packed pages decode byte-equal to
  actual old `kv_store`, including skipped/zero pages and pitched capture/cache guards. It preflights
  captured groups before calling the invariant-trapping store.
- The latter also compares1152 reader cases: two normal/escape/NaN profiles × ratio0/1/2 ×
  heights1/4/16/49 × window3/16/128 × normal+15 forced forms. This includes every reachable scalar,
  decode/vector and WMMA reader, remapped/missing pages, zero keys and mixed source tiles. Existing
  FP4 packed paths independently retain their504 nonzero compressed-key cases. Both the reference
  and optional FP4-fast settings need matched combined gates when requested.

Successful final markers are exactly:

```
PASS ds41_window_codec_exact
PASS window packed producers cases=84 raw=0 byte_errors=0 forms=7 max_M=4096
PASS window packed readers cases=1152 byte_errors=0 forms=32767
PASS ds41_window_packed_paths_exact producers=84 readers=1152 max_M=4096 raw=0 byte_errors=0
```

`ab_window.sh` part `Kvwin` requires `Kvfp4`; `Kvfast` remains independent and requires `Kvfp4`.
Every arm explicitly sets WINDOW_PACKED0/1. The existing packed producer M4096/work bound applies;
unpinned work4097 remains refused. No packed arm can supply the KLD reference. Do not enable any
model arm until all card processes pass zeroRaw/zero byte errors on all four cards, each in fresh
unpinned, EXACT_TALL1, and LCP2048+N1281 processes. No benchmark recommendation follows from the
payload saving; reader/packing overhead must be measured end to end at192GiB, plus the96GiB
functional/spill gate, with off/current-packed/window-packed controls, short/long/prefix/Kh/KLD
and cancellation/concurrency gates. Live-growth composition needs its separate cohort gate.

## Explicit harness authority

Keep `ab_window_window_kv.sh` beside the snapshotted `ab_window.sh`. Its checker paths are
`WINDOW_KV_CODEC_CHECK` and `WINDOW_KV_PATHS_CHECK`. The ordinary `KV_COHORT_MANIFEST` remains
mandatory. The build owner supplies `WINDOW_KV_COHORT_MANIFEST` with exactly these records:

```
ds41-window-kv-cohort-v1
source_commit <same40hex source commit as the ordinary packed cohort>
packed_cohort <SHA256 of the complete ordinary KV_COHORT_MANIFEST>
codec_checker <full SHA256 of WINDOW_KV_CODEC_CHECK>
paths_checker <full SHA256 of WINDOW_KV_PATHS_CHECK>
```

The ordinary manifest binds the served engine, architecture, libref, libr4d and FP4 checker files.
The extension binds the new checkers to that same explicit source authority. Each fresh window
receipt binds both manifests, the full served library hash, fast mode and unique window token.
The gate verifies stopped-QFN state, this controller's stop owner and both occupied research locks
before any checker; inherited masks/tuning/feature flags are cleared in child processes. It rejects
old receipts, changed artifacts, duplicate/missing scopes and any Raw/error/skip. It runs no checker
after model arms begin.

Enabled engine logs must show one startup declaration and one preallocation admission per rank,
then actual store and attention launch markers per rank, all format3/row528/page8448. Existing
format1/2 store/index/attention proofs remain mandatory. Off arms reject any WINDOW_PACKED marker.
These are dispatch/geometry proofs, not numerical closure; the independent card and model byte
gates supply that proof. If the NaN-domain oracle fails, leave this feature disabled and report the
actual failure before choosing a revised format.
