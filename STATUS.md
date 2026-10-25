# Status

This is a research checkpoint from 25 October 2026. It has not been deployed.

| Component | Result |
|---|---|
| Synthetic RAM cache → native owner → GPU expert FFN | Passed on one RX 9070 XT, including default and host-kernel-argument modes, on the preceding combined Core revision. |
| Compact native metadata and source/declaration contract | Passed Release, Address/UndefinedBehaviorSanitizer and ThreadSanitizer CPU checks. |
| Bounded physical source/export/read bridge | Passed all three CPU modes against synthetic complete expert records. |
| Current Foundation v2 Core, real HOST loader and failure cleanup | Corrected fixture v3 passed Release and Address/UndefinedBehaviorSanitizer checks. ThreadSanitizer validation remains pending. |
| Foundation v2 independent source review | Both identified findings were repaired and reviewed. |
| Actual Engine startup through the native cache path | Pending implementation and validation. |
| Full GLM checkpoint, four-GPU inference, quality and performance | Not validated. |

Foundation v2 fixes a routing-profile admission error, reserves backend-global control credit before generic use, and corrects a checker compilation error. The earlier failed build is retained in `evidence/`; later results do not turn that failed revision into a pass.

Fixture v3 completes the test harness's `nothrow` allocation/deallocation hooks. It uses the same Core v2, keeps the same budget assertions and leaves sanitizer diagnostics enabled. The failed fixture-v2 ASan result is included in the evidence.

The next Engine work must keep one checkpoint/source cache, omit native-owned expert payloads from ordinary loading, prove resident uploads complete, and preserve owner/stream lifetimes through failures. The source audit also identifies EOG upload lifetime, Ctx cleanup and HOST virtual-memory accounting requirements.

The current fixtures use a small synthetic model. Passing them does not establish real-model memory use, token throughput, prefill speed, TP4 behavior or MTP support. The requested-allocation budget is not a process RSS limit.

The portable repository entrypoint compiled and passed all seven Release checkers using an existing Core whose source bytes and build profile were verified against the export. A fresh Core build from this repository was not repeated. Original receipts keep their source identities and individual proof scopes.
