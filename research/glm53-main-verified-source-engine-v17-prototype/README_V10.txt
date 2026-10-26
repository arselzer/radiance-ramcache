Core8 / Fixture10 — bounded one-input generation and cache experiment knobs

SOURCE ONLY. New immutable derivative from frozen Core7/Fixture9. No author
compiler/checker/model/GPU/service action. Parent owns every build and runtime.
The Core7 baseline window is a separate gate and is not assumed successful.

Old --transport and --one-token argument counts, default one output, and JSON
schema glm53-actual-verified-main-engine-v1 remain. Internal one_token delegates
to bounded generate(span1). The new explicit manual mode is:
  BUILD/glm53_main_verified_engine_manual --generate MODEL EXPERIMENTAL_HOME COMPLETE_BUNDLE PRODUCER_PROOF TRUSTED_ID OUTPUTS ONE_TOKEN_TEXT [--hot-mib N] [--host-payload-gib N] [--read-chunk-kib 64|1024]
OUTPUTS must be2..16. Exactly ONE tokenized prompt token is still required by the
same actual admission; no chat, multiple prompt tokens, new prefill capacity,
main CLI/API deployment, MTP, kernels/public ABI, or general multi-request API.
Actual Program maxTok1/seq1/spec0/exact/BF16KV/context default512/emptyresidents,
real Q/state/attention blocks, COLD8/maxAcquire8/regions9 remain unchanged.
ignore_eos=true, greedy/temp0/top_k1/seed1 and no speculation remain unchanged.

New schema glm53-actual-verified-main-generate-v1 adds generated token IDs/count,
startup/request elapsed ns, Sink first/last token ns, fixed requested host witness
bytes, actual host StepPlan+RadBatch witnesses, and existing provider counters.
Each output requires one joined step: PREFILL1 at context position0 followed by
DECODE1 at positions1..N-1; decode input must equal the previous committed output.
Witness captures no device token/position pointer. The stepper is the sole witness
writer; harness validates after serve joins it and all ranks. Session successful
complete_frame establishes all four ranks finished exactly42 prepare gates and
sampler/final collect. Last output is sampled, not fed back through another pass.
Fixed16 arrays +Sink32 are explicitly bounded under the existing8MiB named host
control envelope; Sink is drained while waiting and holds the whole bound even
with no consumer. Output caller span commits only after full success.

Quiescent provider statistics are copied from the owned actual sources after
serve joins; they are not simulated. payload_reads counts RECORDS, not syscalls;
payload_read_bytes is buffered pread bytes, not physical NVMe traffic. HOT hits
still run unchanged Owner/Mover checksums and all selected file-generation/stat
checks. No checksum shortcut, algorithm, cache policy or mover fence is changed.
Startup full verification is separate from request timing; no speedup claim.

Options/defaults and named requested host budget:
HOT default64MiB/rank UNCHANGED, explicit0..2048MiB; read chunk default64KiB
UNCHANGED, exact choice64KiB or1MiB. host payload default8GiB UNCHANGED, bounded
<=16GiB; HOT>1GiB requires explicit16GiB. All four HOT+COLD+landing requested
owners plus loader input/output/COL and8MiB controls must fit before allocations:
  host = input_limit + output_limit + column_limit +8MiB
         +4*(HOT +8*3342336 +30228480).
Default2GiB+2GiB loader caps/COL64MiB yield default4.53253173828125GiB requested;
HOT2GiB yields12.28253173828125GiB, admitted only with16GiB envelope. Metadata,
thread stacks/device/headroom have their existing separate named gates. These are
requested conservative envelopes, not RSS or whole-system physical fit. OS page
cache/mapped checkpoint/vendor/driver/stdlib RSS remain excluded. Parent still
requires fresh physical resource gates before a model child.

New CPU-only target, explicit mode:
  BUILD/glm53_main_bounded_decode_check --run-host-decode
Authored3groups: output/cache/preallocation refusals and Sink32 bound; actual HOST
Scheduler+KV+BatchBuilder/commit with synthetic metadata for2 and16 outputs;
actual protocol-only Session PREFILL/DECODE frames through42 gates/all4 ranks.
No manufactured source/DEVICE/Session runtime admission or model correctness.
Old gate, vocabulary checker and37 inherited bodies stay exact. All THREE active
private Core maps are the same Core8; historical maps remain historical.

Parent matching source build (same normal options as Core7):
  cmake -S CORE8 -B CORE_BUILD -DRAD_WITH_HIP=OFF -DCMAKE_BUILD_TYPE=Release -DRAD_BUILD_GLM_TP4_CACHE_EXPERIMENTAL=ON
Use the actual existing Core option spelling from its CMake; no child execution.
Fixture configure/build (parent controls compiler/sanitizers/deadlines):
  cmake -S FIXTURE10 -B CHECK_BUILD -DRADIANCE_SOURCE=CORE8 -DRADIANCE_HOST_BUILD=CORE_BUILD -DCMAKE_BUILD_TYPE=Release
  cmake --build CHECK_BUILD --target glm53_main_bounded_decode_check glm53_main_verified_engine_gate glm53_main_vocabulary_check glm53_main_verified_engine_manual
AQL requires matching whole Core8 archive/private headers and explicit
-DRADIANCE_AQL_CORE=ON. No CTest/auto-run/model/maintenance is added.

No parent build/runtime outcome is claimed in this receipt. Root must validate
Release/ASan/TSan and all inherited regressions, then independently authorize/own
any subsequent GPU window. Core7/Fixture9 and publication7 stay frozen.
