# Status

This is a research checkpoint from 25 October 2026. It has not been deployed.

| Component | Result |
|---|---|
| Synthetic RAM cache → native owner → GPU expert FFN | Passed on one RX 9070 XT, including default and host-kernel-argument modes, on the preceding combined Core revision. |
| Compact native metadata and source/declaration contract | Passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer CPU checks. |
| Bounded physical source/export/read bridge | Passed all three CPU modes against synthetic complete expert records. |
| Current Foundation v2 Core, real HOST loader and failure cleanup | Corrected fixture v3 passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer checks, including preserved Core regressions. |
| Foundation v2 independent source review | Both identified findings were repaired and reviewed. |
| Actual Engine startup through the native cache path | Implementation in progress; validation pending. |
| Standalone four-rank cache preparation coordinator | Passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer: 9 cases and 24 refusal groups. Not yet wired into Engine. |
| Full GLM checkpoint, four-GPU inference, quality and performance | Not validated. |

Foundation v2 fixes a routing-profile admission error, reserves backend-global control credit before generic use, and corrects a checker compilation error. The earlier failed build is retained in `evidence/`; later results do not turn that failed revision into a pass.

Fixture v3 completes the test harness's `nothrow` allocation/deallocation hooks. It uses the same Core v2, keeps the same budget assertions and leaves sanitizer diagnostics enabled. The failed fixture-v2 ASan result is included in the evidence.

The next Engine work must keep one checkpoint/source cache, omit native-owned expert payloads from ordinary loading, prove resident uploads complete, and preserve owner/stream lifetimes through failures. The source audit also identifies EOG upload lifetime, Ctx cleanup and HOST virtual-memory accounting requirements.

The four-rank coordinator compares exact routed expert IDs and waits for each rank's local preparation before allowing expert computation. The first proposed experimental profile terminates the process on a cache error; the HOST subprocess test verifies that termination policy, not GPU recovery. Real GLM geometry, source ownership and Engine/Ctx integration remain required before enabling TP4 caching.

The current fixtures use a small synthetic model. Passing them does not establish real-model memory use, token throughput, prefill speed, TP4 behavior or MTP support. The requested-allocation budget is not a process RSS limit.

The portable repository entrypoint compiled and passed all seven Release checkers using an existing Core whose source bytes and build profile were verified against the export. A fresh Core build from this repository was not repeated. Original receipts keep their source identities and individual proof scopes.

Qwen3.8-Flash-Next with FP8 experts is also under investigation. The official Qwen FP8 checkpoint exists, and three representative source headers confirm E4M3 expert codes with BF16 scales in 128×128 blocks, a format Radiance already supports. The ordinary four-bit `.rad` linked upstream is not an eight-bit expert model despite its filename.

The proposed recipe in `research/qfn-fp8-conversion/` changes only the expert quantization rule and retains the ten protected BF16 experts. It requires the original BF16 checkpoint; expanding the existing four-bit weights cannot recover precision. Its approximately 169 GiB size is an estimate, not a completed conversion or memory-fit result. A separate official-FP8 import route needs the shared PLE scale mapped to each source shard and a global-scale conversion rule. That route is source-reviewed but has not yet passed a bounded converter round-trip test. The reports in `docs/QFN_*` explain both paths and the existing disk-backed n-gram cache option. No full QFN checkpoint was downloaded or converted for this research.

The official-FP8 overlay must explicitly validate the parent scale and every shard alias before conversion. A recipe override can let the stock converter proceed with missing input scales, so a successful converter exit alone does not establish a correct import. The bounded test is being designed to check actual output values and to reject missing or unequal aliases in that preflight.
