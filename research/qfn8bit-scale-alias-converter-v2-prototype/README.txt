Bounded QFN shared-scale alias -> actual rad-convert regression V2 SOURCE ONLY

No author compilation, numerical/import/converter execution, GPU or new model
download. All data are synthetic (<1MiB each fixture, eight small directories).
The frozen real-source scale/header receipts remain separate from these bytes.

Parent commands, actual existing converter/libquant from matching original Core:
  cmake -S HERE -B NEW_BUILD -DRADIANCE_SOURCE=ROOT/radiance-src -DRADIANCE_BUILD=ROOT/build-ds41-aql
  cmake --build NEW_BUILD --parallel 1
  NEW_BUILD/qfn_alias_converter_check --run-host \
    ROOT/build-ds41-aql/bin/rad-convert NEW_BUILD/libqfn_alias_arch.so \
    NEW_BUILD/libqfn_alias_layout.so ROOT/build-ds41-aql/radiance_home/quantizers/libquant.so

This uses the ACTUAL unchanged converter and quantizer, not a copied numerical
implementation. A private converter-only architecture maps3 physical PLE shards
of3/5/7rows*160 plus a gate/up/down FP8 expert. Stock test_import_planes only
describes canonical multi-plane layout; launch returns RAD_E_STATE and is never
called. It is not the production QFN architecture or a native inference reader;
that matching production layout is exercised separately by the metadata planner.
No fake numerical success, model inference, GPU or service deployment is claimed.

The PLE bytes include all254 finite E4M3 encodings, both signed zeros and FP8
subnormals. Exact bytes span uneven/unblock-aligned source boundaries. Parent
BF16[1]0x3951 and3 sidecar alias scalars0x3951 are checked BEFORE each converter
invocation. Expected output code bytes come from the independent physical byte
generator, not a round-trip decoder or observed output. Expert code/scales are
unmatched by the recipe and must copy byte-identically, including separate
gate/up scales in fused logical row order. Input files get an immutable regression
fingerprint before/after (noncryptographic, no hostile writer/security claim).

AUTHORITY CORRECTION: missing/wrong scale refusals are OUR overlay preflight.
Stock converter may accept a quantizer output recipe while source FP8 scales
are missing, then decode unscaled code values. Wrong finite source scales are
also valid quantizer inputs; converter exit0 alone is not precision proof. See
ROOT/QFN_SCALE_RECIPE_BYPASS_CORRECTION.txt. This fixture must never pass a bad
overlay to normal conversion and claim stock refused it.

OFFICIAL SELECTOR CORRECTION
This NEW fixture adds the official source configuration quant_method=fp8,
activation_scheme=dynamic, weight_block_size=[128,128]. The earlier successful
v1 fixture had no quantization_config and did not test architecture selection
for the official descriptor. It remains preserved at the original folder and
its parent execution receipt validation-qfn8bit-scale-alias-20261025T160346Z.

Current QFN plugins register an empty quant descriptor. Every positive converter
argv here explicitly supplies --set quantization_config.quant_method=. The
converter accepts an empty RHS, applies it to Checkpoint::meta(), re-derives
meta.quant, and selects the empty-descriptor architecture. Native source FP8
block128 geometry and code/scale tensors remain unchanged. The synthetic input
config is never edited; the output descriptor must record empty quant. This is
an explicit selector override, not proof that the unmodified official config
works with the plugin. Its original FP8 source metadata must remain recorded
separately in any future provenance receipt. Numerical byte proof is pending
parent execution of THIS v2; do not substitute the v1 result.

Authored exact matrix (execution pending):
group1: actual plan-only, exact decimal conversion, old recipe decimal conversion;
        both converted outputs require exact PLE codes/global scale and expert planes.
group2: seven preflight refusals, no converter invocation/output: all scales absent,
        late scale absent, unequal finite alias, F32 alias, [1,1] alias, wrong global
        scalar and late NaN code. These are bounded fixture guards, not new Core behavior.
group3: two actual converter metadata/recipe refusals: no-global output rule violates
        fixture declaration; invalid zero fixed scale. No completed output may remain.
group4: identical official-style source config WITHOUT the empty-descriptor
        override must fail with the exact architecture selector diagnostic,
        and leave no completed output.
group5: a non-converter child closes both output descriptors then waits; the
        same reap deadline must kill/reap it (1second tiny test).
Total5groups,6 actual converter subprocesses,7 preflight refusals, plus1 tiny
non-converter timeout child. Logs<=64KiB, converter child deadline60s includes
EOF/reaping. SIGKILL is followed by bounded2second reap grace; an OS-stuck child
causes immediate test-process _Exit125. PDEATHSIG is installed before exec.
This is test-process containment, not hardware-fault recovery or a recursive
process-tree kill. Parent still runs its independent process-group deadline.
Private files only, no shell. Complete main-model payload absent.

Future full conversion remains gated on complete source/alias preflight and
actual whole-output checksums. This proof cannot establish all official source
values finite from headers or recover BF16 precision lost in a four-bit file.
The full official-FP8 nonexperts differ from the planner's controlled variant.
Planner multi-sequence KV ceilings mean all sequences at FULLctx simultaneously;
a c8 ceiling miss does not mean stock shared-KV c8 configuration is impossible.
