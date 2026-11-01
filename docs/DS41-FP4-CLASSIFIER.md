# FP4 pointwise classifier — standalone source prototype

2026-11-01, isolated base83ed2f3036b9a49feb22f2bb79f10b9553805f70.
SOURCE ONLY: no compilation, import, syntax/AST, test, simulation, disassembly,
GPU, server, service or model operation by this author. Current53fe, all existing
Ph/kernel/growth cohorts and sealed helpers are untouched. Nothing in serving
code includes or calls this prototype. There is NO active runtime flag or engine
integration. This is a component proof candidate, not a passed byte/performance gate.

## Actual cost and prior exclusions

`libr4d/r4d_ds41_elem.hip::ds_e2m1_rt` still checks seven candidate magnitudes with
rounded FP32 subtract/abs/compare, before the existing scale multiply and BF16
store. It serves native compressed latent, index key and index-query roundtrips,
both ordinary and already-fused chains, including n128 and original-code capture.
Eight index-query operations ×32 heads ×128 columns alone execute32,768 scalar
FP4 quantizations/token/rank; key/latent producers add geometry-dependent work.
This cost is in plain BF16 KV mode too. It is not an estimate of exposed time.

The full retained archive review and newer GPU/CPU/reader audits contain no
implementation or measurement of this seven-distance-loop replacement. Existing
FP4 codec work captures/restores codes AFTER this quantizer; it does not remove
the quantizer search. CPU FP4 weight decoding, dense hardware-FP8 decode, scored
scatter, stream/FMA, Ph, packed-PV pair sharing, n128 norm partition, collective
slices and parked score-fusion/attention-epilogue work are different mechanisms.
No old throughput percentage or lossy requantization result is reused here.

## Closed range and interval proof

The helper consumes raw IEEE FP32 bits and returns exact old E2M1 value bits and
nibble only for FINITE |x|≤6. Refusal leaves output untouched and means execute
the LITERAL ORIGINAL loop. No scale, division, clamp, amax reduction, multiply,
BF16 rounding/store, launch geometry, KV representation or downstream math moves.

For a=|x| the magnitudes are0,.5,1,1.5,2,3,4,6. Their midpoints and winning codes are:

| Midpoint | Below | Exactly at tie | Above |
| --- | ---: | ---: | ---: |
| .25 |0|0|1|
| .75 |1|2|2|
|1.25|2|2|3|
|1.75|3|4|4|
|2.5|4|4|5|
|3.5|5|6|6|
|5|6|6|7|

The old loop minimizes rounded distances, so real-number nearest-neighbor alone
is insufficient. For ANY a in an interval between adjacent POSITIVE magnitudes,
subtraction against BOTH endpoints is exact by Sterbenz: a and either endpoint
lie within a factor2. Thus the whole interval, every neighboring FP32 value and
the exact midpoint retain the original comparisons/even-code tie. Every farther
candidate's real distance exceeds the smaller endpoint distance by at least .5
(the smallest spacing between distinct positive magnitudes). Its FP32 distance
rounding error is below2^-21 throughout this bounded domain, so no farther
candidate can become a spurious winner. Intermediate non-winning ties in the
search cannot change that exact final minimum.

For0 versus .5, if a≥.25, `.5-a` is exact in the relevant interval and compared
with exact `a`. If a<.25, rounded `.5-a` is at least .25 and still strictly greater
than a, including the immediate predecessor of .25. At .25 both are exact and0
wins. Smaller normal/subnormal inputs cannot challenge .5; even if DAZ/FTZ clears
a distance to zero, magnitude0 remains the winner. The sign is copied from the
original input bits, retaining -0 even when a subnormal is flushed.

The helper compares only monotone positive raw-bit thresholds and returns the
old literal magnitude bits plus original sign. All NaNs, infinities and |x|>6
REFUSE, despite the producer normally clamping first. This bound is essential:
for very large x the old subtractions can round ALL distances equal and the final
even candidate4 wins. A blanket saturation-to6 classifier would change bytes.
NaNs run the old `v!=v` return+0; no NaN-payload substitution is invented. The
unchanged scale multiply can itself create NaNs/Inf/subnormals; its exact stored
bits remain independently card-gated, never compared with numeric/NaN equality.

## Source oracles and their exact limits

`ds41_fp4_classifier_test` is a pure host target, with explicit no-fast-math,
no-contraction and rounding-sensitive flags. Five cases check original magnitudes,
signed zero, untouched refusal, huge-value fallback, all seven ties and adjacent
floats, ±4096 raw-ULP neighborhoods, every BF16 image and exponent/mantissa strata,
x86 FTZ/DAZ profiles with restored MXCSR, and unchanged scaled FP32/BF16 bit stores.
The test-only reference retains the original loop. These cases are UNEXECUTED.

`make fp4-classifier-check O=...` builds a standalone HIP tool. Root alone runs it
inside an ordinary stopped/restored-QFN window holding both research locks plus
build.lock, once on EACH actual card. Its CLI requires the actual full libr4d path;
it records the real HIP device/BDF and never claims TP4 from one card's PASS.

The raw pointwise oracle compares full uint32 bits with explicit fast/fallback
counts and guards. Twelve producer cases compare every BF16+padding/guard bit to
the ACTUAL `r4d_rad_quant_roundtrip` export:512/G16/E4 scale,128/G32/pow2 scale,
4096/G32/pow2 scale, four profiles each, exactly65,536 BF16-seeded groups/case.
Inputs include all BF16 images, signed zeros/subnormals, NaN payloads/signs/Inf,
boundary neighbors and source scale255. Scale/amax code in this independent
candidate transcription is checked by that complete actual-producer comparison.
There is no timing, model authority or source-only assertion of GPU NaN closure.
The tool does not gate future fused producers, captured codes/scales, store or
reader composition: those remain mandatory separate checks after runtime wiring.

## Conditional runtime/counter plan, NOT implemented

Only after root's component byte gate: add default-off FP4_CLASSIFIER1 admission
for actual known DS41 FP4 producers, explicit0 in every control. Preserve every
flag0 kernel/call, FP8 path, unsupported geometry and literal old fallback. Cover
plain quant, fused norm/RoPE, fused n128, index-query RoPE and captured main/index
producers; classify the already-computed CLAMPED quotient exactly once and reuse
that same returned code for capture. Keep the old scale/product/store expressions
and all original operation/reduction boundaries. Unknown/tuned/unproved readers
must retain the complete old path; no silent partial fusion or new quantization.

Startup must bind source/library, selected known op families and exact group/scale
geometry on all active ranks. Diagnostic mode needs bounded per-device/per-op
counts of ACTUAL finite-classifier values and old-fallback values, not predicted
counts from shape. Reduce counts within a block, publish one checked increment,
retain per-op/context ownership and read only after normal drain, never per-token
D2H. Charge fixed counter/readback bytes; flag0 does no counter work. Actual>0
fast values for every admitted producer family are a pre-model proof. Price any
ON instrumentation identically in matched diagnostic arms, then separate it from
quiet timing. A host-only flag or standalone oracle is not serving-use evidence.

Fresh same-cohort fused/producer/capture/store/readers must preserve literal codes,
scales, uint16 BF16, zero padding and exceptional products on all four cards under
unpinned/exact/LCP profiles. Then original full3397 direct-old KLD, all replies/
counts/prompt IDs/Kh, original prefix/cancel/recovery,96GiB miss and N1/N4/N8 gates.
No cached-weight/full-RAM assumption, payload allocation, budget change, graph
repartition, hardware denorm-mode change, speed forecast or adoption follows.
