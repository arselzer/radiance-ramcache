# Exact FP8 window-storage candidate, 31 October 2026

Source only, off engine `0f62a578`. No build, host test, card check, emulator,
server, GPU or performance measurement was run by this lane. This commit adds
an isolated candidate codec and strict oracle sources, not native page/store/
reader wiring or an engine enable. Existing packed main/index commits remain
unchanged. Variable budgets, smaller-cache misses and every base reply byte
remain required.

## Existing behavior and history

The native DS41 window is already rounded through E4M3 FP8 with one
power-of-two scale per32 values, then stored as BF16. The combined512-wide
latent is passed to `kv_store` as two256-wide K/V slices; the sparse DS41
attention reader treats the resulting full512 values as its latent row.
Changing this window to FP4 adds loss and is outside this task.

Archive `PREFILL-CRITIQUE.md` explicitly records FP8/FP4 roundtrip-before-BF16
cache writes. `PREFILL-ENGINE-SURVEY.md`'s old exact FP4 lead concerns only
compressed/index rows, not this FP8 window. Its historical float-equality
test was not signed-zero/NaN byte proof; the newer main/index codec supplies
strict producer capture instead. The historical `RESEARCH-3-LOSSY.md` FP8 KV
item estimated little direct decode benefit at short contexts, not a measured
native-window allocation result. Dense FP8 weight windows and VRAM expert
packing are separate, already-tried mechanisms. No exact native FP8-window
physical-storage engine trial was found in the archive/NIGHT references.
Raw captures and every old patch were not exhaustively re-read; the survey's
old external test path was unavailable in this worktree/environment.

Generic `kv_store_fp8` is also different: it rounds BF16 again using constant
K/V descales, whereas this window's original scales vary per32 columns and
row. Reusing it or a generic `--kv-cache-dtype fp8` path does not preserve the
existing DS41 roundtrip bytes.

## Producer-domain proof and remaining card gate

Current `r4d_ds41_elem.hip:block_scale` has a distinct FP8 branch:
`amax=max(amax,1e-4f)`, then `2^ceil_log2(amax*(1/448))` for UE8M0. Because
inputs to this quantizer are already BF16, every finite scale exponent byte
is105..247 (2^-22..2^120); Inf amax gives source byte255=+Inf. NaNs do not win
the group's amax. These are custom producer exponents, not standard UE8M0's
byte255=NaN decoding.

The smallest nonzero E4M3 magnitude is2^-9, so a finite nonzero product is at
least2^-31. Index FP4's source-scale1/subnormal/FTZ tag is therefore ineligible
and unnecessary here. Finite products have at most4 significant bits; integer
dyadic decoding preserves BF16 exactly, including zero sign and overflow.
The source host cases cover every finite code×scale and all65,536 BF16 amax
bit patterns. No BF16 re-quantization or idempotence assumption is used.

Exceptions still require actual card proof. E4M3 NaN codes expand to signed
canonical F32 NaNs before multiplication. The normal decoder proposes the
corresponding canonical BF16 bits, but does NOT treat CPU/GPU NaN payloads as
equivalent. Every pack compares all32 actual producer uint16 values. A group
whose normal decode differs can use a literal pair if it has at most two
distinct uint16 patterns; otherwise it returns `Raw` without writing output.

For source255, dividing finite inputs by+Inf produces signed zero, while
Inf/Inf and NaN inputs can produce signed qNaN codes. Multiplication by+Inf
then yields NaNs. This differs from FP4's NaN-to-+0 quantizer. The card oracle
must establish that the actual resulting patterns fit the literal escape;
there is no CPU NaN-payload assertion, raw sidepool or mid-generation abort
admitted for an active packed window.

`kv_store_bf16` copies uint16 values directly. It does not add the float
load/narrow NaN quieting of some `entry_store` paths. The capture should be
inside the original standalone/fused producer before storage, retaining its
unchanged BF16 result and ORIGINAL FP8 code/scale.

## Isolated candidate format

Header `arch/common/rad_ds41_window_codec.h` uses namespace
`rad_ds41_window_codec`, standalone `Kind::Fp8Pow2G32=3`. Existing core ABI
enums/admission still reject3; this is a reserved candidate, not activation.

Each row is planar `[n original FP8 bytes][n/32 exponent bytes]`, with no
row padding. For n512 it is528B. A group with stored scale0 instead uses its
32-byte code region for two little-endian uint16 patterns followed by32
selector bits; the remaining24 bytes are zero. Zero-filled pages decode+0.
Scale1..104 and248..255 are invalid stored descriptors. Source255 may only
become a captured literal pair, never ordinary stored-scale arithmetic.

`pack_captured_group` returns `Exact` or `Raw`, checking actual uint16 bytes
and preserving output on refusal. `decode_row` validates all scale bytes
before writing output; `decode_row_bits` is the device accessor for already
admitted records. No engine can silently interpret an unsupported record as
a shorter BF16 tensor.

## Runtime/page plan, deliberately unimplemented

1. Extend original-code capture for native FP8/UE8M0 group32 in standalone
   `quant_roundtrip` and fused norm/RoPE paths. The current main/index capture
   accepts FP4 only. Keep its off template and BF16 arithmetic unchanged.
   A transient T×528B capture adds1.03125MiB atC2048 or2.0625MiB atC4096 if
   the buffer plan cannot overlay it; account for that before claiming net
   memory saving. No permanent BF16 cache shadow.
2. Add a separate physical WINDOW contract with logical BF16 shape unchanged.
   Per-layer block bytes are `block_size*528`, rather than FULL's one producer
   row/page rule. Native block_size16 gives8448B/layerpage and337920B across
   40 layers. Store slot s at block=s/16, row=s%16 with that full pitch. Keep
   page tables and sequence/ring index semantics unchanged.
3. Add distinct guarded packed-window store/attention forms. Sparse attention
   has TWO independent caches: window operand1 and compressed operand4.
   Current packed main/index guard intentionally refuses a packed window;
   do not weaken or bypass it. Carry separate formats/pitches and validate
   EVERY actual RAD_KV operand. Decode BF16 bits before each old scalar,
   fast vector/word, WMMA QK and value load, retaining the original FMA,
   reduction, head and selection order. Preserve null/kDsZeroKey handling.
4. Extend logical hash decoding, declaration/page validation and fingerprinted
   format metadata. Existing FULL helpers accept only main/index formats1/2;
   shorter WINDOW U8 views must never reach a plain BF16 reader. Refuse unknown
   aliases, tuned/unproven rows, generic stores, incompatible streamed/ring
   paths and external arbitrary BF16 writes until proven.
5. Audit physical COW/prefix/tiering and window checkpoints independently.
   Their current copies use physical `bytes_per_block`, `layer_bytes_per_block`
   and `layer_stride`; `checkpoint_copy` rotates/names window pages and copies
   one layer fragment per page. That suggests reuse, not a proof. Test wrap,
   partial pages, immutable prefix sharing, COW, restore/cancel, zero pages,
   skipped slots, mixed packed compressed+window and rawF32 tail separation.
   Growing-KV inventory pins WINDOW pools as fixed floors; new physical sizes
   can remain compatible with that policy after the layout tests pass.

At the current C2048 source geometry,138 pages×640KiB=86.25MiB/card becomes
44.47265625MiB of window payload, saving41.77734375MiB before alignment and
new transient/work metadata. AtC4096,266 pages save80.52734375MiB. Page counts,
layers, block size and allocation budgets must remain derived, not hardcoded
from these examples. There is no speed or adopted-resident gain claim.

## Oracle source and required gates

Auto-discovered host target `ds41_window_codec_test` has six source cases:
layout/zero pages, exhaustive finite products, native scale domain, captured
bit checks, exceptional literal pairs and invalid-row refusal.

`make window-codec-check O=...` builds
`ds41_window_codec_device_check LIBR4D.so`. The required final marker is
`PASS ds41_window_codec_exact`. The supplied library is mandatory: the oracle
compares an independent original-code capture transcription against the
actual current `r4d_rad_quant_roundtrip` output. It also exhausts all native
finite code/scale pairs on the card and source255 zero/NaN patterns, and uses
four512-column input profiles covering all65,536 BF16 bit patterns, mixtures
of finite/NaN/Inf/signedzero/subnormal values, pitched BF16/capture rows and
untouched byte guards. Every comparison is uint16 equality and requires
zeroRaw/zero differences. An independent transcription alone cannot pass the
complete oracle. Bind any receipt to full library SHA, binary SHA, source and
guarded GPU-window provenance.

This first oracle does NOT validate future fused capture, physical-window
store, scalar/fast/WMMA readers, captured replay, COW/prefix/snapshot or model
behavior. All require their own byte tests before any runtime arm. Then use
long KLD, exact replies/tokens/KV hashes, warm/resume/window-boundary cases,
N4/N8 and96GiB spill in normal two-lock QFN stop/restore windows. Price extra
decode/capture/store work against the measured memory/resident change.
