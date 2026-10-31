# Allow the existing lent arena to use more than64 slots

Source only. No build, test, syntax check or GPU/model window has run for this change.

The first4K window kept Qe64. Its sizing oracle reports about88 physical native expert slots in the idle activation arena, while the runtime and harness cap the requested arena count at64. That leaves usable space idle. This is a capacity observation, not an access-hit or speed measurement.

Keep the ring maximum64. Raise only the arena maximum128 and the combined scheduler maximum to64+128=192. The existing descending, aligned arena slot carve still clips to the actual protected-level boundary and existing arena size; this does not allocate a larger arena. The original withdrawal, copy-event fencing, grace, failure fallback and publication code is unchanged. No expert arithmetic or reduction changes. Existing unset/0/1..64 values and all defaults keep the same behavior. RAM/expert budgets remain configurable.

Prepared host cases exercise65/88/128 arena admission, strict129/malformed/huge refusals, unchanged ring65 refusal, physical88-slot clipping, combined192 bound, and reload/withdrawal/inflight-copy fences above64. Existing all-rank4096 shape/overrun coverage remains necessary. Root must build and run exact p1_stager_test and ds41_shape_probe_arch_test under the load rules, and syntax-check the harness under the same lock.

After the coherent new build and fresh legacy byte oracles, use matched Lc4096/P1/Pci4096 Qe64/Qe128 arms (all other flags identical). Require actual per-rank counts above64 before measuring this hypothesis; a physical fit of64 or less is no experiment. Compare longKLD to plainM16, logical Kh/complete replies/counts on long/prefix prompts, N4/N8, and96GiB functional spill before adoption. Price additional cold/background reload traffic and full workload time. A win cannot be inferred from merely publishing more residents. Do not combine a first cap measurement with packed-KV/B16/growth or other new switches.
