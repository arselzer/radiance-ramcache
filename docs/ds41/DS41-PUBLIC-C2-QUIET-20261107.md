# Public C2 DSpark quiet performance — 2026-11-07

The current public ABI33 prototype improves single-stream code decode by **15.2%** in a matched ABBA bracket. All 96 replies and original prompt/generated counts match their complete frozen references. This is a measured prototype result; installation, defaults and public refs have not changed.

| Original workload | Plain decode, tok/s | DSpark, tok/s | Change | Evidence |
|---|---:|---:|---:|---|
| Code, 16 requests, cap256, N1 | 24.81 | 28.59 | +15.2% | Two plain endpoints enclosing two DSpark repeats |
| Chat, 16 requests, cap64, N1 | 24.55 | 25.23 | +2.8% | Exploratory single pair |

Rates use total generated tokens divided by total decode time for each arm, then the arithmetic mean of the two same-configuration code arms. The code plain endpoints differ by 0.0128%; the two DSpark repeats differ by 0.983%. The separate canonical mean-rate calculation also passes the predeclared5% endpoint gate (+15.30%); no request, arm, token, cap or observer payment was removed. These code results do not replace the published held-out or concurrent N4/N8 measurements.

All six servers were fresh, completed their original16 requests, stopped with client0/wait0/nonforced receipts, and had distinct process lifetimes. Original numerical native-FIRST checks, complete policy gates and each DSpark draft-origin gate passed. Required query/policy/observer/wait overhead remains paid; raw target/state/timeline capture was disabled for timing. QFN was restored with its unchanged container/image, then answered the normal Paris completion before result interpretation.

Configuration: four selected9070XTs;256GB host RAM/192GiB expert cache; fixed131072 context, N1/C1/p2048, Lc2048/P1/Qe64/Rb/Vp/Cs/Ada/Li64, explicitJ0/Y0/Qpr0. DSpark uses D2v/Ds/Db2/Dx/Dpp and frozen policy v1:0.937915564:0.97501117. Ds still declares max_spec5; Db2 caps actual proposals at2. All original prompt sets/caps and arithmetic/reduction rules are unchanged. The optional R9700 is unused.

The same immutable cohort previously passed the full original finite86-per-arm target/state/functional scope (578 rows,74,723,840 target words,2,647,144,000 live-state bytes,65 seeds, full99987 prompt and long66-token answer). That is fixed131K/N1 finite qualification, not broader concurrent, auxiliary-card, prefix-cache, growing-KV, cancelled-native-state or universal full-state qualification. Chat is not bracketed. No40TPS forecast or automatic adoption follows.

## Exact local authority

- Runtime fbc0f812e82721f36bc2f2f5190f9ad17471355a / tree d2b4793cf0db4bad37a729f9927af283b6106f18.
- Cohort: work/gpu-resume-kv-j1-20261106/cohort-public-c2-root1; external authority SHA5819f2ba0f1f29f6e1e5230137a1b43cf41839128a3a4f314227cf4080fc6e21.
- Source: work/public-c2-quiet-dspark-window-source-20261107, SOURCE101bbde32db042dc5437c1dd03ea7bd7d9a3223c234b59a2da5a48ebce4c3b15.
- Window: ab-20261107T043255Z, winDSparkQuiet131072-20261107T043249Z. CPUprepare76182/chain52463/post all EXIT0.
- Actual result: work/public-c2-quiet-dspark-window-source-20261107/run-root1/root-post-20261107T050805Z/result.json, SHA5df8256f6b10d3195c04fe244a129f18c76e93fda97d13aa95d7aea49b25dfef; status EXACT_96_CALL_PERFORMANCE_SCREEN_PASS, errors[].
- Normal proof in the same post directory: production-normal.json SHAd22f687b533f46959f596485cc44561826841814553b9d901c87c6d017747bd7; QFN PID2350959/ticks38439225, Started2026-11-07T05:07:14.433067549Z.
- Finite prerequisite: work/public33-c2-finite-window-source-20261107/pair-131072-root1/root-post-20261107T032357Z/result.json SHA7498e9aaab0a50ceaf79dbd85c4f0bfd35f56d3c03c911eb4620ca4eaeb6a7dc.

All work paths above are relative to /home/as/ds41-claude-build/. Complete raw results, failed histories and exact authorities remain local and unchanged.
