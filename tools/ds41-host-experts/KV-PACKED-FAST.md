# Optional packed-cache integer reader

This is source for a separate experiment, **not a replacement for the initial
packed KV implementation or its proof gates**. The original
`arch/common/rad_ds41_kv_codec.h` remains unchanged and is both the default reader
and the independent reference. The new `rad_ds41_kv_fast.h` changes integer BF16
bit reconstruction only. Producer arithmetic/capture, page layout, scales,
literal escapes, physical allocation, prefix ownership and all consuming
FMA/reduction/WMMA arithmetic remain unchanged.

Both switches must be explicit for a fast arm:

```
RADIANCE_EXPERIMENTAL_DS41_KV_PACKED=1
RADIANCE_EXPERIMENTAL_DS41_KV_PACKED_FAST=1
```

FAST unset/`0` uses the existing codec. FAST other than `0`/`1`, or FAST=`1`
without PACKED=`1`, refuses architecture startup. Direct packed library calls
also reject malformed FAST values. The library reads the option once per
process; compare modes in separate server/checker processes. There is no
automatic replacement after any test. No cache capacity is hardcoded.

Source-supported cost changes, without a speed forecast:

- For E4M3/group16, every finite scale/code product is BF16-normal and needs at
  most eight significant bits. One nonzero integer CLZ replaces the reference
  dyadic decoder's find-most-significant-bit loop. Scale0 takes the literal path.
- For power-of-two/group32, code magnitudes2..7 advance by64 in BF16's positive
  encoding. An exponent offset replaces the dyadic decoder entirely, with
  explicit source1/half subnormal, stored254/half FTZ, signed zero and overflow
  cases. Stored255 remains invalid; no Inf-product NaN is recreated.
- The native width/format is specialized to512/16 or128/32. Eight-value loads
  share one scale/metadata check and prepared coefficient/exponent. Pair loads
  share both literal words and one selector byte when aligned. Generic eight
  value spans crossing a group retain exact scalar reconstruction.

The runtime variant adds selection branches/internal argument state. Their cost,
register pressure, code size and interaction with existing load overlap are
unmeasured. A shorter source path does not establish a throughput improvement.
The original codec still performs all store-side validation; FAST never weakens
the closed-producer-domain/zero-Raw prerequisite.

Prepared proofs (not executed by the source lane):

1. Host `ds41_kv_fast_test`:131072 normal scale/code-byte combinations, every
   uint16 literal in both pair slots with random selectors/padding, mixed
   unaligned rows, scalar/eight-value reads at every starting column, and
   malformed kind/scale/index/width/null/late-row metadata. Rejecting a row must
   preserve the entire output. Existing codec tests remain independent.
2. Target `kv-fast-check` builds `ds41_kv_fast_device_check`, which compares the
   unchanged reference against FAST on the same GPU over the same domains.
   Expected receipt: normal131072, pair131072, rows256, metadata1, bad0. This
   binary performs no benchmark. Every GPU invocation belongs inside the usual
   stopped/restored-QFN two-lock window; every build/test uses build.lock.
3. After original packed gates pass, rerun `ds41_kv_packed_paths_check LIBR4D.so`
   with FAST=`0` and FAST=`1` in separate processes, both unpinned and under the
   adopted LCP/N128 profile. This covers all reachable scalar, fast-decode and
   WMMA reader forms, including mixed window/packed transpose tiles. All bytes
   must match; unsupported calls are failures.
4. Only then price reference-codec vs FAST arms on the current192GiB setup:
   long KLD/Kh and identical replies/counts, prefix continuation and N4/N8,
   with the same formats, source/library cohort and model settings. Preserve a
   smaller-cache/miss regression arm. No performance claim or adoption without
   a clean paired measurement. Keep a PACKED=`0` base in the overall series.

Startup prints `KV_PACKED ... fast=0|1`. Successfully enqueued normal packed
index/attention reader markers contain `decoder=reference|integer-fast`, once
per rank/role. Require the requested decoder marker on every attending rank;
markers alone do not prove completion or byte identity.
