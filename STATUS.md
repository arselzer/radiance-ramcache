# Status

Current source is Core V8 (1,535 files), Engine fixture V10, Components V14 and archived Runner V9. It passed one controlled test with a one-token raw prompt and two sampled outputs, including one actual recurrent decode. It is a bounded research checkpoint, with no production deployment, ordinary chat, quality or performance claim.

| Core V8 stage | Result |
|---|---|
| Matching HOST Release/ASan/TSan full regressions | Passed 20 commands per mode with exact Core V8/fixture V10. |
| Focused suites in all three HOST modes | Passed vocabulary mask/actual CPU vocabulary and bounded-decode checks; three commands per mode. |
| Actual CPU vocabulary guard | Passed model154880/tokenizer154856/padded24/local38720 with backend calls0. |
| Matching AQL Core and Components V14 builds | Passed; the controlled trial below exercised the actual model. |
| Actual metadata and complete-data prewalk | Passed all four rank bindings/native plans, 161,715,585,024 native bytes and 5,088 ordinary jobs. Metadata/prewalk itself grants no Engine capability and launches no model kernels. |
| Private combined four-GPU trial | Passed raw `a`/ID64 -> `[154822,154822]`, one PREFILL1 and one DECODE1 frame, context512/spec0/seq1/exact wire. |
| Retirement and QFN restoration | Passed healthy GLM teardown, same-container restoration with a 40.663-second stable interval, and an independent four-token QFN completion. |
| Fresh public fourteen-program build/runtime | Passed a fresh whole Core V8 build from the published source and all fourteen explicit HOST programs. No private Core archive was reused. |

The full [validation summary](evidence/glm53-main-verified-v8/VALIDATION_SUMMARY.json) links original receipts/logs. All build/runtime/service/GPU actions were performed by the parent. This export author performed source/hash/AST work only. The preceding aborted stop is preserved separately: no GLM child ran in that earlier attempt, and the same QFN container was restored.

The actual trial used the defaults: 64MiB HOT/rank, 64KiB read chunks, 8GiB named host-payload envelope, eight transient expert consumers/rank and zero fixed residents. It admitted exactly the real 154856 tokenizer IDs while retaining physical model/logit geometry154880 and masking24 tail rows. Both frames completed42 cache-prepare gates on all four ranks; frame1 consumed the committed frame0 output. Actual quantum/attention-block/state-chunk remained4/4/1. No batch field or mathematical geometry was rewritten.

Recorded startup was832.387s, including full source verification/loading. The two model/sampler/collect frames were5.102s and4.596s; total request time was9.722s. Those single-run durations do not establish throughput or a performance improvement. Each rank's quiescent provider reported672 misses/record reads, zero HOT hits, and2,246,049,792 buffered payload-read bytes across both frames. Record-read counters are not syscall counts or physical NVMe traffic. This checkpoint still uploads selected experts to transient GPU consumers and adds no persistent GPU retention or checksum bypass.

Code bounds generation through16 outputs, and HOST tests exercise2/16-output Scheduler+KV commits and protocol metadata. Only2 outputs were tested on GPUs. HOT2GiB/1MiB read-chunk options are explicit bounded experiment controls, not measured gains. Arbitrary multi-token prompts, normal prefill, chat templates, useful text, long generation, 262K, MTP, quality, multi-request operation, GPU/driver recovery and production serving remain unvalidated. Conditional long-context arithmetic is not runtime support.

The thirteen retained portable checker bodies are unchanged, with source-identical paths now selected from fixtureV10/componentsV14. The new fourteenth target is exact `decode_checker.cpp`, linked to the same matching Core and existing wrap flags. Default portable commands build only; explicit HOST execution never opens real checkpoint/native-store/tokenizer data or invokes archived Runner/maintenance code. Compiler/ABI/environment/ccache/source-home/cache guards are retained. New versioned build directories prevent implicit reuse of Core V7 headers or archives.

Core V7 remains fully preserved at `d04b562` / manifest `ea0b8f81`: its one-input/one-output trial and fresh public thirteen-checker result retain their original scope. Restore the four files under `research/historical-core-v7-overlay` onto a copy of current `radiance/` and verify all1535 original V7 pins. Existing V6 and V5 overlays apply afterward. The archived base manifest and seven overwritten wrapper/docs files reconstruct the prior complete public checkpoint. No old raw package, evidence or source map is rewritten.

Earlier research and QFN conversion/placement notes remain under `docs/` and in [the exact preceding status](docs/provenance/base-d04b562-checkpoint/STATUS.md). Their conditional metadata, synthetic, single-record, CPU or private-archive scopes remain separate from the current two-output GPU evidence. The inherited synchronous fresh-state-clear refusal limitation remains; this checkpoint adds no general hardware-fault recovery.

The fresh public build/runtime tested manifest `b0fef123` (3,550 files), archived with the seven overwritten metadata files under `docs/provenance/portable-tested-b0fef123-checkpoint/`. The final delta adds only evidence and documentation; compiled source, public wrappers and checker bodies are unchanged. Exact receipts and logs are linked by [the fresh-build summary](evidence/glm53-main-verified-v8/FRESH_PUBLIC_BUILD_SUMMARY.json).
