# Status

Current selection: Core V6 (1,535 files), Engine fixture V8, Components V12 and the archived Runner V4. Production deployment is not included.

| Current stage | Result |
|---|---|
| Matching Core and experimental module builds | HOST and AQL Release builds passed; matching sanitizer builds supplied the HOST suites below. |
| HOST regression suites | Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer passed 20 commands each, including the same production source validator and retained regressions. |
| Actual complete-data prewalk | Passed all four source-rank bindings, NativeInventory bindings and final Native plans; prepared 5,088 ordinary jobs after 193,536 native-owned views were omitted. No loader execution, Engine or model kernels. |
| Portable twelve-program Release build | Passed against a source-identical existing Core at the original `d2915f6e` source checkpoint. Programs compiled; the separate runtime result below also passed. |
| Current portable twelve-program execution | Passed all twelve HOST programs against the 3,021-file `f0eaee18` snapshot. Its exact manifest is preserved; final changes add evidence/docs with identical source/tools/CMake. |
| Four-GPU Engine startup and one-token output | Failed before source admission: `RAD_E_FORMAT`, `ColdAdmitted`, tokenizer154856 versus padded model154880. No GLM token or model computation. |
| Original QFN restoration | Passed same-container restoration, 40+ seconds of stable health and an independent four-token completion. |
| Normal prompts, 262K, MTP, quality and throughput | Not validated. |

Original receipts and all 20 logs per HOST mode are under [the current evidence](evidence/glm53-main-verified-v6/CPU_PREWALK_SUMMARY.json). Earlier Core V5 results retain their original manifests and scope.

The actual prewalk took 1,046.67 seconds and passed 82,370 checks with 27 intended refusals. It verified 161,715,585,024 native bytes/source-rank cuts and omitted the same expert payload from ordinary loading. Its 5,088 remaining jobs comprise 2,788 identity, 1,292 widening and 1,008 relayout jobs. The reusable input/output/column reservation is 389,021,696 bytes and job metadata is 13,326,920 bytes. These requested reservations are not process RSS or a full Engine memory-fit result. [The exact prewalk output](evidence/glm53-main-verified-v6/complete-prewalk/result.json) explicitly records `loader_execute_called:false` and `runtime_ready:false`.

Core V6 fixes an actual plain-source scale handling error: CheckpointSources keeps a null scale placeholder beside every unscaled piece. The production validator now requires those nulls for exact canonical plain encoding, and requires physical nonnull scales for affine encoding. It preserves source bounds and transform exclusions. Fixture V7 corrected only a boolean test sentinel; fixture V8 synchronized the stale nested strictP3 Core pins. All three current active pin maps equal Core V6. Earlier failed source/build/prewalk receipts remain unchanged.

The first manual profile remains one raw prompt token and one sampled output, maxTok1/seq1/spec0/exact wire, context512 and no fixed resident experts. The private request cannot be enabled through ordinary CLI/HTTP serving. Core V6 currently fails its strict vocabulary-size guard before source admission; the passing prewalk does not repair that Engine constraint. Even a later one-token success would not establish arbitrary prompts, 262K, quality, speed or deployment. Runner V4 is manual server-specific source; fresh V4 preparation and original private gates are required, and it is never invoked by portable tools.

The actual window enumerated four gfx1201 devices and loaded the checkpoint/tokenizer. It then returned -14 (`RAD_E_FORMAT`) in phase `ColdAdmitted`: 154,856 tokenizer entries do not equal the model's 154,880 padded rows. The snapshot reports source_admitted=false, native_bytes_verified=0, ordinary_jobs=0, ordinary_upload_bytes=0 and transport_ready=false. No GLM inference or memory-fit result is established. [The exact failed child and restoration evidence](evidence/glm53-main-verified-v6/GPU_WINDOW_SUMMARY.json) are retained.

The same original QFN container was restored; health stayed stable for over 40 seconds. An independent check verified the same full ID/image/name and generated four tokens on QFN, not GLM. The earlier pre-stop health timeout had issued no stop and launched no child. This window is closed, with no further GPU run queued. A separate padded-vocabulary/mask design is future work; frozen Core V6 still contains the failing guard.

The inherited synchronous fresh-state-clear refusal limitation remains documented. GPU/driver recovery is not claimed.

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
| Full GLM four-GPU inference, quality and performance | Not validated. |

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
