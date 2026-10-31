# P1 threshold boundary oracle — SOURCE ONLY

No runtime/kernel arithmetic, defaults or admission rules change. This isolated
follow-up starts atf209d87d and only extends the standalone MXFP4 checker and
existing host threshold cases. No build/test/import/syntax/device work was run.

Current P1_MIN_ROWS already accepts17..4096;128/320/512 are legal. Both pass and
consumer-cut layer admission require at least that many rows and always at least
17. Layers of16 or fewer retain the CPU path. Qe64 keeps MISSES≥C safety for any
unarmed/mixed pass; no reader/loan guard is relaxed. Host cases now explicitly
include128/512 in parsing, card-layer admission, CPU bound and consumer-cut grids.

The existing `p1` mode keeps its behavior, timing loops and PASS marker. Its
native row list omits128/320/512 and their immediate neighbors; MMAX128 does not
add128, and its final Ms.back print is not tested-height coverage. Its projection
comparison also explicitly permits different NaN payloads. Do not reinterpret
older P1 receipts as a strict uint16 boundary oracle.

New optional CLI: `ds41_mxfp4_device_check LIBR4D.so p1-threshold 2048 ROUTES`.
It sets the same LCP2048/P1 pins as ordinary P1, requires a bound real-route file
(bounded shape≤43 layers/65536 rows and64MiB, no truncated/trailing bytes), and
tests explicit heights17/33/48/64/100/127/128/129/256/319/320/321/511/512/513/
1024/2048. Real rank0/rank2 plus skewed/uniform/hot routings use native96-expert
GU4608×5120/DN5120×2304 block/packed/skipped tables. GPU-vs16 and forced-split
comparisons use literal uint16 equality INCLUDING NaN payloads; any difference
fails. Existing host NaN-payload diagnostics stay separately labeled, never a
host byte-equality assertion. Skips/tail zeros, pitched/guarded SwiGLU views,
full/span wave composition and small-cache/last-expert/no-local cases remain.

The new mode extends its own wave composition to1..4 and requires at least one
actual four-expert wave containing live rows. Ordinary mode still uses1..3.
It returns after numerical coverage; no old performance sweep runs. Expect116
numerical cases with required real routes and all listed heights. Its coverage
line reports actual rows/cases/bad/device-NaN-payload differences/live-wave4;
the final required marker is:

`PASS: P1_THRESHOLD_BYTE C=2048 min_rows=128,320,512 boundary_heights=9 gpu_uint16=exact skipped=untouched tail=zeros waves=1-4 pitched=guards real_routes=1`

Run only under the ordinary two-lock stopped-QFN/restored-answer window, with
the actual current checker/library SHA bound, fresh ordinary LCP2048/sweeps,
balance/P1 and CPU SwiGLU/chain prerequisites. The threshold model gate must
refuse a missing/failed new mode, including a synthetic NaN-payload difference;
never weaken the comparator to make it pass. Model correctness still requires
each threshold's full3397 KLD records against the approved old plain base plus
all unchanged long8K/harness replies/counts/prompt IDs/logical Kh. Source legality
or a synthetic card proof does not establish a useful threshold or a speedup.
