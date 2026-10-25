# Status

This is a research checkpoint from 25 October 2026. It has not been deployed.

| Component | Result |
|---|---|
| Synthetic RAM cache → native owner → GPU expert FFN | Passed on one RX 9070 XT, including default and host-kernel-argument modes, on the preceding combined Core revision. |
| Compact native metadata and source/declaration contract | Passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer CPU checks. |
| Bounded physical source/export/read bridge | Passed all three CPU modes against synthetic complete expert records. |
| Current Foundation v2 Core, real HOST loader and failure cleanup | Corrected fixture v3 passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer checks, including preserved Core regressions. |
| Foundation v2 independent source review | Both identified findings were repaired and reviewed. |
| Actual Engine startup through the native cache path | Transport v4 passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer, including retained regressions. Closed tiny HOST fixture; no model step or serving. |
| Standalone four-rank cache preparation coordinator | Base and bounded prefill-lookahead variants passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer. The latter has 11 cases and 27 refusal groups. Not yet wired into Engine. |
| Actual GLM checkpoint and TP4 expert layouts | All 72 source files verified; metadata resolved supported GPU kernels and canonical layouts for all 12,096 main experts per rank. No GPU execution. |
| One complete actual GLM expert across four logical ranks | Passed source-to-record and independent byte comparisons, Reader ownership and late-failure checks. One expert, not a complete native store. |
| Full GLM four-GPU inference, quality and performance | Not validated. |

Foundation v2 fixes a routing-profile admission error, reserves backend-global control credit before generic use, and corrects a checker compilation error. The earlier failed build is retained in `evidence/`; later results do not turn that failed revision into a pass.

Fixture v3 completes the test harness's `nothrow` allocation/deallocation hooks. It uses the same Core v2, keeps the same budget assertions and leaves sanitizer diagnostics enabled. The failed fixture-v2 ASan result is included in the evidence.

Transport v4 uses one checkpoint/source cache, omits native-owned expert payloads from ordinary loading, checks resident data and preserves owner/stream lifetimes through failure and retirement. Its six Engine groups include a narrow fix for an empty KV sentinel that legitimately has a non-null reserved base. The earlier failed v2 run remains in the evidence. Actual GLM source binding and four-rank execution are the next integration steps.

The four-rank coordinator compares exact routed expert IDs and waits for each rank's local preparation before allowing expert computation. The first proposed experimental profile terminates the process on a cache error; the HOST subprocess test verifies that termination policy, not GPU recovery. Real GLM geometry is now checked; complete source ownership and Engine/Ctx integration remain required before enabling TP4 caching.

The current fixtures use a small synthetic model. Passing them does not establish real-model memory use, token throughput, prefill speed, TP4 behavior or MTP support. The requested-allocation budget is not a process RSS limit.

The updated portable entrypoint compiled and passed all eight Release checkers using a verified source-identical transport-v4 Core. A fresh Core build from this exported repository was not repeated. The earlier seven-check Foundation run and the initial guard-only invocation failure remain recorded with their original scopes.

The real GLM stock placement query is tight even at 512 tokens, one sequence and MTP disabled. The smallest checked fitting pool is 26.5 GiB per rank with an assumed 512 MiB runtime allowance per GPU. Including an illustrative 8 GiB host margin gives 114.0625 GiB, slightly above the estimated 113.6 GiB available after stopping production; that margin can partly overlap memory already excluded by the estimate. This neither proves startup failure nor establishes a comfortable fit. The bounded cache remains useful for longer contexts and host headroom.

At 262K context, GLM's BF16 KV needs about 2.876 GiB per GPU. None of the checked 26–28 GiB/rank stock host pools fits all weights; even 28 GiB/rank leaves 1.21–2.78 GiB unplaced per rank under the stated runtime allowances. See [the actual-checkpoint report](docs/GLM53_REAL_CHECKPOINT_PREFLIGHT_20261025.txt) for the assumptions, real-record proof and remaining full-store work.

Qwen3.8-Flash-Next with FP8 experts is also under investigation. The official Qwen FP8 checkpoint exists, and three representative source headers confirm E4M3 expert codes with BF16 scales in 128×128 blocks, a format Radiance already supports. The ordinary four-bit `.rad` linked upstream is not an eight-bit expert model despite its filename.

The BF16-source proposal in `research/qfn-fp8-conversion/` changes only expert quantization and retains ten BF16 exceptions. A separate hybrid recipe now comes from the actual production container's recorded recipe: it removes only the generic expert GPTQ rule, imports official native FP8 experts and retains the other 25 rule lines exactly. The bounded hybrid test passed 651 checks, including exact retained and new plane equality. Full-model strict reuse coverage, metadata compatibility and byte verification remain required; expanding existing four-bit weights cannot recover precision.

The hypothetical expert-only FP8 upgrade passed actual stock native-layout and placement queries for one 262,144-token sequence, including MTP depth 3, with disk-backed n-grams and 20 GiB host pool per rank. This quotes about 88.196 GiB host memory including the stated allowances. The plan retains today's other weights, disables prefix caching and uses deterministic placement; it is not a fit result for the whole official FP8 checkpoint or for eight full-length sessions. Runtime margins and available RAM after stopping production are assumptions. Startup, loading peaks, quality and speed remain untested.

The official-FP8 overlay explicitly validates the parent scale and every shard alias. A recipe override can let the stock converter proceed with missing input scales, so exit success alone does not establish a correct import. The five-group converter fixture passed 350 checks, including all 254 finite FP8 codes, exact expert planes and the required empty model-selector override. The overlay builder passed eight small CPU tests; it has not run on the full official checkpoint. No full QFN checkpoint was downloaded or converted.

See [the validated QFN preparation report](docs/QFN_FP8_VALIDATED_PREFLIGHT_20261025.txt) for source links, conversion choices, commands, memory assumptions and evidence. Archived stock-Core CMake fixtures require their original matching Core; the Python overlay tests use only the standard library.
