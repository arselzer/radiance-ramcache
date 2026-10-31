# Bank readback complement words — SOURCE ONLY

This isolated prototype starts at5fb651c173b98da4bc805951defc8c6475d72bb3.
No build, test, import, syntax check, emulator, GPU or service action was run by
this lane. Current0348519d query sources/home and every frozen runner/evidence
file remain untouched. The switch is default OFF and has no performance claim.

## Measured reason and prior attempts

The actual successful `bank-packed-fill-retry2/window192-timed-retry2` model
window used5fb, the original1792-token workload, two paid cold/warm cycles and
strict restoration. Its `196608-lend-packed.server.log` records these exclusive
COHORT buckets (all four ranks together, not per-rank measurements):

| Restore | complement ns | H2D ns | D2H ns | full memcmp ns |
|---|---:|---:|---:|---:|
| first |1,879,167,468 |235,422,366 |273,349,659 |259,776,534 |
| second cumulative delta |1,877,107,337 |230,537,724 |269,239,957 |257,353,467 |

Each row covers the same5,414,584,320 raw bytes,288 full-pair reads/writes;
mask invariants have their separate unchanged buckets. Capture complement was
1,875,598,152ns for5,414,590,464B; the dependent capture FNV was5,434,783,695ns
and is unchanged here. File-lender complements were similarly~1.88s. Pinned
allocation and first D2H varied substantially between processes; no attribution
of the entire cold-start delta to complement stores is justified.

`COST-AUDIT.md` first identified serial FNV and restore work without subphase
measurements. `COPY-COALESCING-FEASIBILITY.md` retained strict complement/full
memcmp and identified~612 restore copies, but explicitly awaited cost data.
Those reports and `DIRECT-PACKED-LOAN-FEASIBILITY.md` are under
`work/autonomous-20261031/bank-packed-fill/`. Copy coalescing, lists, extra
readback buffers and direct packed publication remain separate, unimplemented
leads. Existing packed CPU/staticVp/P1 transport or prior neutral fetch-list/
ring/TBO experiments do not establish a complement-store win. No previous
bank-complement implementation or measured alternative was found in that
archive/source history.

Even impossible removal of this ENTIRE restore-complement bucket alone leaves
the measured packed lender~0.66s cold/~0.58s warm above static for this short
workload (63.656/61.449s versus61.121/58.987s). This subtraction is only a loose
counterfactual ceiling inside the already-paid workload, never a prediction or
a separately additive saving. Native copy volume and immutable backup remain.

## Exact boundary

`RADIANCE_EXPERIMENTAL_DS41_DSPARK_BANK_LEND_COMPLEMENT_WORDS=1` requires
`DSPARK_BANK_LEND=1` plus its existing explicit budget/admission. Absent/0 keeps
the literal original forward byte loop. Empty/01/true/negative values refuse.
The engine parses once during existing bank startup and emits one enabled-only
`DSBANKCOMPLEMENT startup` receipt; no steady-query path reads the environment.

The enabled helper complements unaligned uint64 words through constant-size
`memcpy`, then the original byte tail. Bitwise NOT changes exactly the same
eight byte patterns on either endian order, including signed-zero/NaN encodings,
padding and zero tails. It introduces no floating-point work, CPU ISA dispatch,
vector overread, alignment requirement, threads or payload buffer. Every overlap
(including exact alias) falls back to the original forward byte semantics;
loading eight bytes first must not change a partially overlapping dependency.
Zero bytes allows null pointers; negative, nonempty-null and overflowing address
ranges refuse before a load. Image uses its already-validated separate snapshot
and pair relay. The generic helper does not assert allocation ownership.

The same existing complement timer surrounds the work. FNV seed/iteration/order,
copy API/stream/rank/span order, waits, requested bytes, strict literal memcmp,
masks, fixed bank addresses and all-rank readiness/failure latch are unchanged.
Every unread byte still disagrees with its own immutable saved byte. No partial
restore, sampled verification, hash authority or mask repair is introduced.

Enabled mode records checked cumulative `word_calls/word_bytes` only after an
actual successful word loop; scalar tails/overlap fallbacks have distinct
`byte_calls/byte_bytes`. Empty and sub-word spans cannot fake word use. Native
capture/restore emit enabled-only `DSBANKCOMPLEMENT event` status/readiness/
cohort receipts after the same wrapper invariant checks. Counts must reconcile
to the unchanged three complement timing buckets. Native pair/mask sizes are
all multiples of8, so successful native mode requires zero scalar bytes and
positive word coverage of EVERY readback byte. Counter overflow poisons readiness.

All retained `sizeof(Image)`/controller metadata remains charged. Enabled mode
also conservatively charges8192B for bounded transient word/address/index/call
locals and enabled-only receipt formatting (including the existing4096B logger
buffer) BEFORE pinned allocation. There is no new pinned/pageable payload or
additional relay, and no automatic cache-budget expansion or RAM-size premise.
Future unbacked expert-cache and configured headroom reservations are unchanged.
Compiler codegen still needs review: the old loop may already vectorize, or
cacheability of host-pinned stores may dominate. Word source alone proves no gain.

## Required root validation

Host targets are exact `draft_bank_backup_test` and `draft_bank_lend_test`, under
build.lock with GPU labels excluded. New backup cases cover all byte values,
unaligned source/destination offsets, lengths around8/16/32/64/128 and256, zero/
null/invalid address ranges, both overlap directions, untouched guards, exact
old/new copy order/hash/coverage,8192B accounting, and successful-but-short
capture/H2D/D2H faults on every rank. Config cases preserve existing callers and
prove strict opt-in/dependency behavior. No timing threshold is a host assertion.

The existing `ds41_bank_backup_check` CLI and PASS lines remain usable unchanged.
Root must run a NEW fresh four-real-card gate with its additional explicit
`--complement-words` flag (and `--timings` for the actually timed branch) inside
the normal two-lock stopped-QFN/restored-answer window, before model enable.
Require ordinary exit0, original backup/timing PASS and the additional
`PASS: DSBANK_COMPLEMENT_CARD ranks=4 ...` marker. The optional proof captures/
restores old and new images sequentially through the real IO, compares their
entire native rank/direction/device-offset/bytes call sequence with a bounded
charged ledger, identical immutable FNV and full-byte readback. It retains only
one snapshot/relay cohort at a time. It then performs a real successful-but-short
last-card D2H: the seeded zero tail must be caught by full memcmp and poison the
cohort permanently. Original short-H2D/partial-fill/mask proofs also still run.
The enabled-only oracle ledger/locals are separately charged, never an engine
workspace expansion. Guarded N24 matches the measured bank; N1/N32 boundary
coverage remains useful with a sufficient explicit cap.

Before performance interpretation, inspect actual optimized code for both
loops: no new out-of-line memcpy per word, no unwanted spills/overreads, and no
assumption that old scalar source meant scalar machine code. Then compare
fresh matched packed-lender words0/1 on the exact frozen1792 cold+warm workload,
with SAME timing flag and all other controls, plus an ordinary/static reference.
Keep actual all-rank packed hits, full strict restore/query-resume, output/count/
ID/Kh equality, unforced serving exit and normal QFN answer. Report cumulative
bucket DELTAS as cohort ns and whole paid workload/process lifecycle separately.
A96GiB follow-up must observe real miss/refusal/file fallback; a budget label
alone is insufficient. Keep every original prompt set unchanged and do not
enable/adopt or claim a net burst win from this source prototype/card gate.
