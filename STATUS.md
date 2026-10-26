# Status

Candidate source: Core V7 (1,535 files), Engine fixture V9, Components V13 and archived Runner V5. The prior public checkpoint is commit `aacccd8`, Core V6. This is a validated one-input/one-output research checkpoint, with no production deployment.

| Current Core V7 stage | Result |
|---|---|
| HOST Release build and full regressions | Passed; 20 commands with matching Core V7/fixture V9. |
| Focused vocabulary/mask suite | Passed actual HOST mask/top-k rows, grammar intersection, cached greedy refresh, default control and refusals. |
| Actual CPU vocabulary preflight | Passed the same Engine metadata/tokenizer/header guard: model154880, tokenizer154856, padded24, local38720, backend calls0. No Engine start/source admission. |
| AQL Core/module and Components V13 builds | Passed matching compilation; the separate controlled trial below exercised one model step. |
| Fresh V5 tokenizer/HOME preparation | Passed; one raw input token and matching private six-library HOME, no model/GPU execution. No HOME binaries are exported. |
| Core V7 fast actual metadata | Passed all four logical rank graphs with active valid-token profile. No kernel initialization/launch or source readiness. |
| Matching ASan/TSan builds and runtime | Passed all 20 HOST commands per mode. |
| Current complete-data prewalk | Passed all four rank bindings/native plans and 5,088 ordinary jobs; 161,715,585,024 expert bytes. No kernels or Engine admission in this prewalk. |
| Portable thirteen-program build/runtime | Passed Release from a fresh whole Core built from published source at exact manifest `4d9c9e4a`, 3,287 files; all 13 programs ran against that fresh archive. Earlier `8b966d37` reuse evidence is preserved. |
| Core V7 four-GPU startup/one-token trial | Passed combined startup/transport/one model step: one input token, one output ID154822 `[gMASK]`; healthy teardown and same-QFN restoration independently checked. |

Exact original receipts/logs and the preserved focused-ASan executor failure are under [the current validation evidence](evidence/glm53-main-verified-v7/VALIDATION_SUMMARY.json). The first focused ASan attempt printed all test assertions passed but then failed LeakSanitizer under ptrace; it remains failed. The same source/build passed both focused modes on the host rerun, without changing sanitizer diagnostics or source. The parent owns all compiler/runtime/service/GPU actions; this export author performed source/hash/AST checks only.

Core V7 preserves physical model/logit geometry (154,880 rows, 38,720 per TP4 rank) and admits exactly the real 154,856-token prefix after the actual Engine-owned Checkpoint, tokenizer/view and BF16 embedding/head header checks. The existing mask stage excludes 24 tail IDs; it intersects grammar masks and refreshes greedy staging. A zero opt-in preserves stock/default/tiny behavior. Model arithmetic, public ABI27, kernel schema, expert records and cache policy are unchanged.

The focused tests prove CPU staging and existing HOST mask/top-k behavior; they do not simulate a successful TP collective or model forward. The real CPU mode loads headers/tokenizer metadata through the same Engine guard, not start(). The current fast metadata/prewalk and actual trial use active Sampler valid_token_count154856; no old unmasked Core V6 result is substituted. The controlled trial loaded and drained all 5,088 ordinary jobs (16,921,544,160 uploaded bytes), verified 161,715,585,024 native expert bytes and compared 185,066,521,464 mapped source bytes. Actual quantum/attention block/state chunk were4/4/1. The combined child took840.326seconds including full verification/loading/startup/one step/healthy retirement; this establishes no tok/s measurement. QFN restoration passed a40.631-second stable health interval and an independent four-token QFN completion.

The first manual profile remains one prompt/output token, maxTok1/seq1/spec0/exact wire/context512 and no fixed resident experts. Ordinary CLI/HTTP cannot manufacture its private source request. Subsequent recurrent decode, arbitrary prompts, 262K, MTP, full-model numerical quality, speed and production serving remain unvalidated. `[gMASK]` is a valid sampled ID, not evidence of useful text quality. Runner V5 is archived manual source only, with fresh preparation, current focused/prewalk/three-mode gates and the unchanged stop/restore/resource/signal machinery.

The prior public Core V6 checkpoint is retained unchanged: all three HOST modes, full prewalk and twelve portable HOST programs passed, but its GPU startup failed at the exact-equality vocabulary guard in ColdAdmitted. No GLM token or model computation occurred; the same QFN container was restored and independently generated four tokens. Those passes/failure retain their tested manifests and do not establish a Core V7 result.

Core V6 reconstruction overlays the six original files onto a copy of current radiance/ and checks the original1,535pins. V5 reconstruction then applies its two original loader files. All raw archived packages, historical evidence and tested manifests remain byte-identical. The inherited synchronous fresh-state-clear refusal limitation and absence of GPU/driver recovery remain explicit.

The following notes describe earlier individually pinned research stages; they retain their historical scope.

This is a research checkpoint from 25 October 2026. It has not been deployed.

| Component | Result |
|---|---|
| Synthetic RAM cache → native owner → GPU expert FFN | Passed on one RX 9070 XT, including default and host-kernel-argument modes, on the preceding combined Core revision. |
| Compact native metadata and source/declaration contract | Passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer CPU checks. |
| Bounded physical source/export/read bridge | Passed all three CPU modes against synthetic complete expert records. |
| Earlier Foundation v2 Core, real HOST loader and failure cleanup | Corrected fixture v3 passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer checks, including preserved Core regressions. |
| Foundation v2 independent source review | Both identified findings were repaired and reviewed. |
| Actual Engine startup through the native cache path | Transport v4 passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer, including retained regressions. Closed tiny HOST fixture; no model step or serving. |
| Standalone four-rank cache preparation coordinator | Base and bounded prefill-lookahead variants passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer. The latter has 11 cases and 27 refusal groups. At that earlier revision, experimental Engine wiring existed while full-source admission stayed closed. |
| Actual GLM checkpoint and TP4 expert layouts | All 72 source files verified; metadata resolved supported GPU kernels and canonical layouts for all 12,096 main experts per rank. No GPU execution. |
| One complete actual GLM expert across four logical ranks | Passed source-to-record and independent byte comparisons, Reader ownership and late-failure checks. One expert, not a complete native store. |
| Complete actual GLM main-expert store | All 48,384 rank records in 168 shards passed independent comparisons covering every one of the 161,715,585,024 native bytes. Model payloads remain private. |
| Persisted complete-store reader and RAM-cache provider | Actual complete-data check passed across four logical ranks, including all native checksums/source cuts, cache hits without payload rereads, and leases after source/graph destruction. HOST refusals/provider cases also passed all three modes. No Engine admission or GPU model execution. |
| Earlier TP4 Core, preparation protocol and upload completion | Fixture v4 passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer, including retained regressions. Ten portable Release checkers also passed against the matching existing Core. |
| At the earlier stages: full GLM four-GPU inference, quality and performance | Not validated then. Current Core V7 adds only the single-input/single-output trial above; quality and performance remain unvalidated. |

Foundation v2 fixes a routing-profile admission error, reserves backend-global control credit before generic use, and corrects a checker compilation error. The earlier failed build is retained in `evidence/`; later results do not turn that failed revision into a pass.

Fixture v3 completes the test harness's `nothrow` allocation/deallocation hooks. It uses the same Core v2, keeps the same budget assertions and leaves sanitizer diagnostics enabled. The failed fixture-v2 ASan result is included in the evidence.

Transport v4 uses one checkpoint/source cache, omits native-owned expert payloads from ordinary loading, checks resident data and preserves owner/stream lifetimes through failure and retirement. Its six Engine groups include a narrow fix for an empty KV sentinel that legitimately has a non-null reserved base. The earlier failed v2 run remains in the evidence. Actual GLM source binding and four-rank execution were separate remaining steps for that tiny checkpoint.

The four-rank coordinator compares exact routed expert IDs and waits for each rank's local preparation before allowing expert computation. The first proposed experimental profile terminates the process on a cache error; the HOST subprocess test verifies that termination policy, not GPU recovery. At that checkpoint real GLM geometry was checked; complete source ownership and Engine/Ctx integration were still separate gates. The preceding V5 implementation passed HOST checks; it does not establish current GPU execution.

The retained tiny fixtures use a small synthetic model. Passing them does not establish real-model memory use, token throughput, prefill speed, TP4 behavior or MTP support. The requested-allocation budget is not a process RSS limit.

The preceding portable entrypoint compiled and passed all ten Release checkers using a verified source-identical TP4 Core build. This run reused the existing Core archives; it rebuilt the portable fixtures. A fresh Core build from this exported repository was not repeated. Earlier eight-check transport and seven-check Foundation runs, and the initial guard-only invocation failure, retain their original scopes.

The real GLM stock placement query is tight even at 512 tokens, one sequence and MTP disabled. The smallest checked fitting pool is 26.5 GiB per rank with an assumed 512 MiB runtime allowance per GPU. Including an illustrative 8 GiB host margin gives 114.0625 GiB, slightly above the estimated 113.6 GiB available after stopping production; that margin can partly overlap memory already excluded by the estimate. This neither proves startup failure nor establishes a comfortable fit. The bounded cache remains useful for longer contexts and host headroom.

At 262K context, GLM's BF16 KV needs about 2.876 GiB per GPU. None of the checked 26–28 GiB/rank stock host pools fits all weights; even 28 GiB/rank leaves 1.21–2.78 GiB unplaced per rank under the stated runtime allowances. See [the actual-checkpoint report](docs/GLM53_REAL_CHECKPOINT_PREFLIGHT_20261025.txt) for the assumptions, real-record proof and remaining full-store work.

Qwen3.8-Flash-Next with FP8 experts is also under investigation. The official Qwen FP8 checkpoint exists, and three representative source headers confirm E4M3 expert codes with BF16 scales in 128×128 blocks, a format Radiance already supports. The ordinary four-bit `.rad` linked upstream is not an eight-bit expert model despite its filename.

The BF16-source proposal in `research/qfn-fp8-conversion/` changes only expert quantization and retains ten BF16 exceptions. A separate hybrid recipe now comes from the actual production container's recorded recipe: it removes only the generic expert GPTQ rule, imports official native FP8 experts and retains the other 25 rule lines exactly. The bounded hybrid test passed 651 checks, including exact retained and new plane equality. Full-model strict reuse coverage, metadata compatibility and byte verification remain required; expanding existing four-bit weights cannot recover precision.

The hypothetical expert-only FP8 upgrade passed actual stock native-layout and placement queries for one 262,144-token sequence, including MTP depth 3, with disk-backed n-grams and 20 GiB host pool per rank. This quotes about 88.196 GiB host memory including the stated allowances. The plan retains today's other weights, disables prefix caching and uses deterministic placement; it is not a fit result for the whole official FP8 checkpoint or for eight full-length sessions. Runtime margins and available RAM after stopping production are assumptions. Startup, loading peaks, quality and speed remain untested.

The official-FP8 overlay explicitly validates the parent scale and every shard alias. A recipe override can let the stock converter proceed with missing input scales, so exit success alone does not establish a correct import. The five-group converter fixture passed 350 checks, including all 254 finite FP8 codes, exact expert planes and the required empty model-selector override. The overlay builder passed eight small CPU tests; it has not run on the full official checkpoint. No full QFN checkpoint was downloaded or converted.

See [the validated QFN preparation report](docs/QFN_FP8_VALIDATED_PREFLIGHT_20261025.txt) for source links, conversion choices, commands, memory assumptions and evidence. Archived stock-Core CMake fixtures require their original matching Core; the Python overlay tests use only the standard library.

A separate [single-source FP8 conversion candidate](docs/QFN_SINGLE_SOURCE_FP8_CONVERSION_ALTERNATIVE.txt) keeps the official checkpoint's native experts and re-quantizes its nonexpert weights with the same 15 nonexpert rules. It uses no weight reuse and removes the ten protected-BF16 casts. This avoids mixing checkpoint releases, but no longer isolates an expert-only change to today's model. The recipe is prepared and source-reviewed; full-model conversion, placement and quality remain untested.

The preceding 1,517-file Core included the first GLM main TP4 cache integration: real route hooks, shared rank coordination, checked upload completion and separate model/sampler phases. In that revision full-source activation remained hard closed before declaration and model allocation. Fixture v4 passed the new protocol and transfer-fence checks plus retained regressions in Release and both sanitizer modes. The earlier checker compilation and timing-assumption failures are preserved with their original results; the Core protocol was unchanged by those harness repairs.

The complete-store assembler passed its bounded actual-weight test: two complete experts across four logical ranks, with an independent byte oracle and late-failure injection. A fresh complete source proof hashed all 72 checkpoint files (185,107,699,217 bytes). The full main-expert store then passed assembly, checksums and the independent all-byte oracle: 48,384 records, 168 shards, 150.609375 GiB, about 24.4 minutes. Ordinary weights remain in the original checkpoint; this is not a complete standalone model package or an inference result. Private checkpoint and native-store payloads are excluded from this repository.

The separately archived `glm53-verified-main-source-prototype` reopens strict complete data, binds actual rank geometry and supplies owned native bytes through bounded RAM caches. Its source review closed. All three HOST modes passed 19 reader refusals and 14 preserved synthetic provider cases. The separate actual complete-data check then passed: every native byte was checksum-verified, 185,066,521,464 mapped source bytes were compared with physical source files, and all four rank bindings hashed their exact source cuts. Each rank served eight cache misses and four deduplicated HOT hits, with no further payload reads on the repeated request. Retained expert bytes survived destruction of the sources, graphs and checkpoint, and their accounting tokens retired afterward.

That data run took about 16.7 minutes and launched no kernels or Engine. It used one RX 9070 XT for metadata selection and four logical rank layouts. Requested reader peak metadata was 233,195,840 bytes; observed process peak RSS was 18,153,771,008 bytes including resident source mappings. These are different measurements, and neither is a full-model startup fit result. The eager verification cost is explicit, not a fast-start claim. See [the complete-data result](evidence/glm53-verified-main-source/complete-data/complete-reader-result.json). Original CMake/pin scripts retain original-workspace paths, like the other archived packages.

The selected manual path inherits stock code that ignores a synchronous fresh-state-clear refusal. Normal successful clears are ordered on the same compute streams; complete negative/error-propagation and GPU/driver recovery are not claimed. This staging has not altered that path. No production deployment or serving configuration change is included.
