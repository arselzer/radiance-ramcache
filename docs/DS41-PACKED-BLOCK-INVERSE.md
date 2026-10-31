# Packed CPU cache to original BLOCK bytes

Source-only follow-up off `b089fde0`, independent of the first bank-lending
window. No build, host test, benchmark, emulator, model/weight read or GPU work
was run. This commit adds a pure inverse and byte-oracle source; it changes no
Mover, Engine, RingLender, bank controller, cache layout or runtime enable.

## Why the byte inverse is closed

`ds41_host_experts.h` defines the existing HXP2 packed-cache format. It retains
the release's entire planar code-byte region unchanged. Per32-block round,
each e8m0 byte is either `e0+two_bit_code` or the original byte from the indexed
raw-round region. `packed_scales_equal` already compares every reconstructed
byte against original source scales before cache publication. This includes
source exponent bytes0 and255; decoding them to floating point is unnecessary.

`pk_round`'s layout is explicit for full and partial rounds. A full round uses
eight packed bytes with block j in byte j%8 at shift2*(j/8). A last round of t
blocks uses b=ceil(t/4) bytes, byte j%b and shift2*(j/b). The matrix's exception
list is ordered by `row*ceil((K/32)/32)+round`, each raw record has32 slots,
and only the valid t bytes of its tail round are used. Native down K2304 has
72 blocks, hence two full rounds plus an8-block tail. GU K5120 has160 blocks,
five full rounds.

After recovering these scale BYTES, the existing integer
`ds41_mxfp4_pack_sse2::pack` interleaves them with the original code bytes in
the same17-byte `block_mxfp4` layout as the loader. The nibble permutation maps
planar element2i/2i+1 packing to GGML byte j's low element j and high element
16+j. On non-SSE2 hosts, the existing libquant `pack_of(MXFP4)` is the same
integer relayout fallback. No quantization, dequantization, float operations,
BF16 conversion or new math is introduced. Code bits, including zero codes,
are preserved even where numerical equality would fail to distinguish them.

Native matrix byte counts are exactly
`4608*(5120/32)*17 = 12,533,760` for GU and
`5120*(2304/32)*17 = 6,266,880` for DN. The admitted standard unit's raw pair
at offsets0/12,533,760 is18,800,640 bytes, with no unknown inter-weight gap.
The target is the loader's stored native BLOCK form, not the source file's
planar representation. Original packed-slot headers/alignment/unused exception
capacity are metadata, not missing bytes of the target matrix.

## Pure helper contract

`core/place/ds41_packed_inverse.{h,cpp}` exports
`rad::ds41_packed_inverse::block_bytes(N,K)` and
`to_block(source,source_bytes,N,K,dst,capacity,stats=nullptr)`.

It allocates nothing and uses one32-byte stack scale buffer. Existing
`packed_view` checks the exact header/shape/cap/section sizes. The inverse
additionally validates the ENTIRE active exception list before writing:
strictly increasing, no duplicates and every index in range. `packed_view`
alone does not establish those invariants. Pointer/capacity overflow and
source/destination/Stats aliases are refused. All refusals leave destination
and Stats unchanged; success writes every matrix byte and no capacity tail.
Extra source-slot padding is ignored. Caller owns a valid immutable lease
until materialization completes; there is no internal cache lock or epoch
inference.

## What was already tried

- Packed CPU scales and direct packed CPU kernels were implemented and
  measured in the earlier128GB era. The archive/NIGHT records the paired
  packed/second-chance windows and exact scale-publication gate. Those gains
  are not a forecast for current256GB/192GiB lending.
- VRAM Vp directly consumes a compatible packed form. P1 tall transport also
  accepts packed RAM bytes directly and marks the table entry, including the
  tail/raw-round byte checks. It does not need this inverse.
- PHASE1-PLAN explicitly preferred direct packed transport; a separate
  on-card packed→BLOCK widening kernel was only a fallback if the packed
  twin failed. That proposal was not a measured ordinary-BLOCK reload path.
- RingLender currently only copies borrowed ordinary BLOCK cache hits; a
  packed hit fails `!hu.packed` and goes through original source planes plus
  `Mover::relay_unit_block`. The first bank controller similarly uses source
  extents for packed hits. This follow-up supplies the missing pure inverse,
  while retaining those runtime paths for the first measured bank window.
- Prior staging/free-list and reload experiments concern RAM pressure,
  bandwidth and grace policy. They do not establish this byte inverse's
  end-to-end benefit, and do not justify adding allocations or cache budget.

## Later runtime hook, after the first bank result

Hold both `Mover::HostUnit` leases with touch=false. Validate actual unit,
Program/Plan/source epoch and native GU/DN `ggml_mxfp4` shapes/offsets, source
slot lengths and raw stored byte counts. A packed lease is not ordinary
BLOCK memory and must never be memcpy'd to that target. Invoke this inverse
for both matrices into the already-budgeted pair relay at actual MoveUnit
offsets, then release the leases after CPU materialization. The existing
synchronous target copy and all-rank publication/rollback/restore protocol
remain unchanged. Clearing the entire target pair before filling retains
the existing padding policy; unsupported larger units/gaps need their own
proof rather than inheriting the native-pair assertion.

A cache miss or a previously refused packed unit still reads original source
planes and relays as today. A malformed published packed image must fail
closed under the transition's error policy. Do not allocate a second full
scale plane, shadow pair, worker queue or read buffer. Report inverse RAM
bytes/time distinctly from original-file/relay bytes, fill, strict restore
and complete E2E timing. No data transport saving or performance improvement
is measured by these sources. Preserve96/128/192GiB cache budgets and miss
paths; never infer that packed-cache coverage is complete.

## Oracle sources and required gates

The auto-discovered host target `ds41_packed_inverse_test` has six source
cases. Its independent expected bytes come from the EXISTING libquant
MXFP4 relayout, then the output is unpacked through the existing inverse and
compared to every original code/scale byte. Cases include all256 exponent
bytes and every code byte, all32 full/tail geometries across row boundaries,
raw full/tail rounds, full native GU/DN shapes with512/320 budgets, unaligned
source/destination, untouched guards, malformed header/truncation/capacity,
duplicate/unsorted/out-of-range indices, and source/destination/Stats aliases.
`packed_scales_equal` checks each valid fixture before the new inverse.

After host validation, a runtime adapter test must cover a real packed cache
borrow, both matrix orderings/actual offsets, lease release, miss/refused
fallback and full target bytes before publication. Real-card bank gating then
compares RAM-inverse fills against original-file BLOCK fills, strict readback,
mask/alias invariants and partial-failure rollback. Model windows require
exact replies/tokens across solo→concurrent→solo/cancel/repeat plus reduced
cache misses. Existing bank receipts are not proof of this future hook.
