# Status

Current source is frozen Core V6 (1,535 files), Engine fixture V8 and Components
V12. Matching V6 compilation, HOST/sanitizer checks, portable12, actual full
prewalk and four-GPU startup/one-token results remain pending. No GPU inference
or serving readiness is claimed. CoreV5 passes below retain their earlier scope.

V10 complete prewalk passed all four source-rank bindings, NativeInventory
bindings and final Native plans, then ordinary-loader prepare refused -18.
V11’s cheap actual metadata diagnostic reported 1,067 physical refusals per rank,
all null scale placeholders on valid plain sources; the first was BF16 embedding.
All other reported materializer/layout/destination/share refusal counts were zero.
CoreV6 preserves exact scale/piece cardinality, accepts only null placeholders
for canonical plain one-plane tensors, and requires valid physical scales for
affine two-plane tensors. Production and the current diagnostic share one pure
validator. New HOST source cases are authored; their pass is not inferred. Historical fixtureV6 used an invalid boolean sentinel; fixtureV7 uses a valid aggregate and byte snapshot with identical cases. CoreV6 is unchanged. FixtureV8 then repins the stale nested strictP3 map; all active current maps equal CoreV6, with C++ cases unchanged. The initial failed HOST configuration is preserved.

The exact two replaced V5 loader files are archived as a historical overlay;
all other 1,533 Core files are identical, retaining reconstructible old pins.
V5/V10 packages, receipts and prior snapshot manifests remain unchanged.

The following section records the preceding V5 publication checkpoint.

The selected source is frozen Core V5: exactly 1,535 files, with Engine
fixture V5 and current component fixture V10 archived unchanged. Component V7
remains historical. Core V5 passed the matching Release, Address/UndefinedBehavior
Sanitizer and ThreadSanitizer HOST suites, including retained regressions. The
initial V5 export passed all twelve portable Release checkers using an existing
source-identical Core build. That evidence binds its original 914c311b snapshot;
this updated V10 archive/source map awaits its own portable rerun.

V7's actual complete prewalk failed before Engine after source verification. Its
helper published metadata pointers from a local owner; the retained-owner repair
is in V10. V9 then failed a fast metadata check because an invented 4096 KV ceiling
rejected valid actual flattened metadata. V10 removes that ceiling while checking
exact nonnegative count, vector lengths and owner pointers. V10 AQL fixture
compilation passed. V10 full prewalk subsequently reached ordinary loading, as recorded above. The controlled four-GPU
startup/one-token test remain pending in this publication; no model readiness,
quality, speed or GPU inference success is inferred from HOST/build evidence.

The manual real-source factory exists in this Core. It requires verified
complete data and a private experimental request; ordinary CLI/HTTP serving
cannot activate it. Its first profile is one prompt token and one output token,
maxTok1/seq1/spec0, exact wire, context512 and no fixed resident experts. Even a
later successful run would not validate normal prompts, 262K, quality or speed.
The archived runner is manual server-specific source and has no default action.

The following results describe earlier, individually pinned source revisions.
They remain unchanged and retain their original scope; selected V5 checks are described above.

This is a research checkpoint from 25 October 2026. It has not been deployed.

| Component | Result |
|---|---|
| Synthetic RAM cache → native owner → GPU expert FFN | Passed on one RX 9070 XT, including default and host-kernel-argument modes, on the preceding combined Core revision. |
| Compact native metadata and source/declaration contract | Passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer CPU checks. |
| Bounded physical source/export/read bridge | Passed all three CPU modes against synthetic complete expert records. |
| Earlier Foundation v2 Core, real HOST loader and failure cleanup | Corrected fixture v3 passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer checks, including preserved Core regressions. |
| Foundation v2 independent source review | Both identified findings were repaired and reviewed. |
| Actual Engine startup through the native cache path | Transport v4 passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer, including retained regressions. Closed tiny HOST fixture; no model step or serving. |
| Standalone four-rank cache preparation coordinator | Base and bounded prefill-lookahead variants passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer. The latter has 11 cases and 27 refusal groups. The selected Core now includes experimental Engine wiring; full-source admission stays closed. |
| Actual GLM checkpoint and TP4 expert layouts | All 72 source files verified; metadata resolved supported GPU kernels and canonical layouts for all 12,096 main experts per rank. No GPU execution. |
| One complete actual GLM expert across four logical ranks | Passed source-to-record and independent byte comparisons, Reader ownership and late-failure checks. One expert, not a complete native store. |
| Complete actual GLM main-expert store | All 48,384 rank records in 168 shards passed independent comparisons covering every one of the 161,715,585,024 native bytes. Model payloads remain private. |
| Persisted complete-store reader and RAM-cache provider | Actual complete-data check passed across four logical ranks, including all native checksums/source cuts, cache hits without payload rereads, and leases after source/graph destruction. HOST refusals/provider cases also passed all three modes. No Engine admission or GPU model execution. |
| Current TP4 Core, preparation protocol and upload completion | Fixture v4 passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer, including retained regressions. Ten portable Release checkers also passed against the matching existing Core. |
| Full GLM four-GPU inference, quality and performance | Not validated. |

Foundation v2 fixes a routing-profile admission error, reserves backend-global control credit before generic use, and corrects a checker compilation error. The earlier failed build is retained in `evidence/`; later results do not turn that failed revision into a pass.

Fixture v3 completes the test harness's `nothrow` allocation/deallocation hooks. It uses the same Core v2, keeps the same budget assertions and leaves sanitizer diagnostics enabled. The failed fixture-v2 ASan result is included in the evidence.

Transport v4 uses one checkpoint/source cache, omits native-owned expert payloads from ordinary loading, checks resident data and preserves owner/stream lifetimes through failure and retirement. Its six Engine groups include a narrow fix for an empty KV sentinel that legitimately has a non-null reserved base. The earlier failed v2 run remains in the evidence. Actual GLM source binding and four-rank execution were separate remaining steps for that tiny checkpoint.

The four-rank coordinator compares exact routed expert IDs and waits for each rank's local preparation before allowing expert computation. The first proposed experimental profile terminates the process on a cache error; the HOST subprocess test verifies that termination policy, not GPU recovery. At that checkpoint real GLM geometry was checked; complete source ownership and Engine/Ctx integration were still separate gates. The selected V5 manual implementation is described above; its HOST checks passed, while actual model execution remains pending.

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
