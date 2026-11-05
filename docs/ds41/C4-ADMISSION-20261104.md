# 4K chunks with admit-all: no adoption

The current recommended 2K configuration remains unchanged. The new 4K screen
shows no demonstrated prefill improvement and slower post-prefill concurrent
decode. Its whole-window qualification also fails one urgent-prefetch activity
guard; that failure is retained.

Window `ab-20261104T195802Z` used the unchanged coherent stable engine and all five
stable providers, both homes explicitly `home-stable`, four primary XTs, 192 GiB,
admit-all/Li64, Rb/Vp/P1/Cs, fixed BF16 KV and Lm131072. No new model kernel or
weight conversion was used. Eleven arms cover the original 6.8K/48K/100K prompts
and a quiet C2/C4/C4/C2 sequence, each quiet arm running original48K then the
original held-out eight on the same server. Both research locks were held;
no heavy CPU work or publication ran during the GPU measurements.

All fresh required native gates passed at C4096/P14096 under the unchanged
checker contract. The wrapper exited0. Production QFN was restored normally at
**20:39:36 UTC**, same container/image, exact ` Paris. The capital` completion.

| Observed quiet interval | C2 mean | C4 mean | Change / qualification |
|---|---:|---:|---|
| 48K TTFT | 66.0484s | 66.0272s | −0.032%; control drift−8.91%, timing ineligible |
| Subsequent N8 canonical aggregate rate | 47.1205 tok/s | 43.8280 tok/s | −6.987%; control drift−1.97% |
| Subsequent N8 whole wave wall | 14.0261s | 15.0328s | +7.178%; control drift+0.278% |

Canonical aggregate rate is8×mean of the server's per-request
`timings.predicted_per_second`. The latter two intervals have stable control
brackets, but are descriptive results within a globally unqualified screen;
they do not grant adoption. No result is pooled with older windows.

The strict postcheck returned1 because the C2/Kh/6.8K arm enabled Pu without
consuming an actual urgent rest-read. Its original `.unmeasured.json`, exact
guard message and failed strict summary remain. A separate ROOT recovery checks
all **43 original reply-byte/count records** and **six full prefill digests**,
including that retained unmeasured arm. All match the qualified references.
The recovery explicitly sets `window_qualification_pass=false`, `adoption=false`
and `performance_adoption_qualified=false`; it does not waive the activity gate
or promote any unmeasured timing.

The structural logs provide a concrete lead:

- P1 ring-copy submissions fall from **125,968 to71,400**, −43.319%, identically
  in both orders. These are enqueue counters, not completed-DMA or critical-path
  measurements. Copy volume alone therefore cannot forecast a TTFT gain.
- C2 static residents per rank are618/627/627/627; C4 holds538/547/547/547,
  losing80 on every card. The larger arena lends64 slots instead of48/47/47/47.
  All those slots are resident by shutdown in both orders. Net resident capacity
  is consequently63–64 experts/card lower under C4.
- The lower resident capacity supports memory placement as a decode-slowdown
  explanation, but this is not a separate causal isolation experiment.

Next prefill work should measure actual reuse and critical-path costs with the
bounded copy ledger before funding larger tiles or a layer-major executor.
Further C4 work needs a new mechanism addressing its memory cost, not a repeat
of the same configuration. DSpark qualification and bounded overlap remain
the active decode lanes.

The automatic post-report was delayed by a ROOT launcher bug: a broad `pgrep -f`
matched its own parent command text containing `ab_window.sh`. GPU measurement
and normal production restoration had already finished. ROOT terminated only
the owned waiting chain, ran the unchanged strict post under build.lock, and
retained the raw failure plus separate recovery. A new actual-argv process
helper passes eight launcher/source-text contracts; the repaired future chain
requires a new frozen input manifest. The original frozen chain/pins remain
unchanged. This was a reporting delay, not extra GPU downtime.

Evidence:

- `work/c4-admission/post-root-manual1.log` and original output
  `C4-admission-summary.json` — strict FAIL
- `work/c4-admission/RAW-OBSERVED-RECOVERY.json` — complete raw observations,
  explicit global qualification/adoption false
- `work/c4-admission/STRUCTURAL-EVIDENCE.json` — all four quiet source-log hashes
- `work/c4-admission/POST-WAIT-RECOVERY.json`
- `work/c4-admission/inputs.sha256` and `inputs-after-recovery.log`
- `work/winC4admission-20261104T195732Z.log`
