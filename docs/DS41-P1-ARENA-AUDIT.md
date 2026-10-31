# Optional P1 arena use observation

SOURCE ONLY. No build, tests, syntax check or device/model execution in this
lane. This is evidence collection, not a residency/math change or a speed claim.

`RADIANCE_EXPERIMENTAL_DS41_P1_LEND_AUDIT=1` is default off. P1 startup reads it
once, after its ordinary tunables, and accepts only exact1 with nonzero arena
lending and ring lending0. Unset/empty/exact0 leave the original path and report
unchanged; other spellings refuse. The enabled comparisons must state1 in BOTH
Qe64/Qe128 profiles and include its observation cost.

`ArenaUseAudit` retains fixed128-slot metadata and two-word publication/selection
bitsets inside the existing CardExpertPath owner. It allocates no buffer, event,
worker, device operation or timer. Actual owner `sizeof` grows by the fixed
observer size in both profiles. Existing Qe128 grants can still allocate more
per-slot HostUnit metadata/events than Qe64; this observer does not authorize
larger VRAM or pinned reload payload. The existing arena and two reload buffers
remain the payload sources.

Initialization binds each admitted arena slot to its exact unit and physical
address. Successful observed publication advances a host generation and sets
the published bit only after Mover's actual resident/base agrees. Withdrawal
advances the generation and clears live ownership. At the two existing routed
counts paths, after synchronized counts have passed their ordinary validity
check, an actual ring-resident skip reports its selected row count. The observer
counts it only when the exact arena unit/base, live publication generation and
current RingLender resident state agree. Zero selected rows produce no use claim.
Unknown/stale ownership poisons only the diagnostic validity bit; it changes no
production return, skip, table, copy, arithmetic or scheduling action. No env
read occurs on this path. Selection lookup is bounded by the admitted slot count.

The existing single shutdown P1 stats line per rank gains this suffix ONLY when
enabled:

```text
P1_ARENA_USE_AUDIT_V1 enabled=1 valid=1 granted_slots=N granted_bytes=B published_distinct=P published_upper=PU selected_distinct=S selected_upper=SU selections=T metadata_bytes=Z
```

These are actual lifetime distinct slot counts, not reload attempts. `upper`
means arena index≥64, matching the first64 top-down slots in both Qe profiles;
ring slots are excluded. A Qe128 extra-capacity gate requires a valid all-rank
receipt, grant>64, correct exact byte product, positive upper publication AND
upper selection. More than64 distinct selected slots is useful additional
evidence, not a requirement for demonstrating one actual extra slot used.
Qe64 must report no upper publication/selection. A flag, rank0 grant log, total
reload count or final residents count alone cannot substitute. No slot is a
new expert-cache budget, and no reader is allowed while the arena owns it.

Exact host target `p1_stager_test` adds six SOURCE cases for default-off/strict
configuration, disabled observations, distinct and upper selection, publication
generations/withdrawal, wrong owner/address/count and invalid/duplicate geometry.
Root must build/run the target under build.lock before admitting any model gate;
fresh LCP4096/balance/P1 card proofs and the independent long KLD/reply/Kh gate
still apply. The current389bf676 cohort lacks this observer, so the prepared
arena gate refuses with `INCOMPLETE_USAGE_GATE` rather than claiming usage.
