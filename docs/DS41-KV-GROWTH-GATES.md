# Growing-KV card and model gates

Prepared against integration381d30ad; source only. No build, syntax check, host test, card
invocation or model window has run for this checker. The live source status is documented in
[DS41-KV-GROWTH-LIVE.md](DS41-KV-GROWTH-LIVE.md). This gate changes no engine arithmetic or mode.

## Fresh physical backing proof

Build `ds41-kv-backing-check` with the same core as the candidate server. Invoke only inside
`work/gpu_cmd_window.sh` while both research locks are held and production QFN is stopped:

```
<build>/bin/ds41-kv-backing-check --inside-guarded-window
```

The new tool refuses host fallback, non-gfx1201 cards, a visible-card count other than4,
unsupported VMem or differing granules. Its reservation is16 actual backend granules/card,
with64MiB additional free-card headroom required. A granule must be a positive whole MiB and
at most64MiB. This bounded fixture uses real device-allocator Pools, deferred KVManager
carving, the actual KVLiveController and actual KVManagerBackingPublisher. It does not run
packed producer/decoder kernels; raw byte patterns deliberately test backing lifetimes.

It requires startup physical backing below maximum; allocation-driven all-rank growth;
published candidate IDs backed on every layer and rank; stable base addresses and layer
strides; a retained prefix reference pinning backing after sequence release; and return to
the initial physical total after that reference is released. Every held raw-state/window/
prefix byte is copied H2D, read back and compared exactly through shrink. Scratch is two64KiB
pinned buffers. All copies finish on private streams before mapping transactions.

A second transaction uses actual map/unmap operations with an auditing wrapper: all four
free heaps must be withdrawn before any mapping operation, and all four actual target maps
must exist before the first target publication. Then rank3 maps a real new granule and the
wrapper reports an injected error. Actual maps must return to the recorded old maps; no
target publication is permitted. Finally the wrapper injects a rollback-unmap refusal,
leaving a real physical extent in the ledger: every publisher must be poisoned and expose
no allocatable IDs. These are faults at the backend API seam, not induced driver faults or
proof of hardware access-protection failure. The host inaccessible-map/access-restore cases
remain mandatory. The actual Engine additionally treats even a repaired mapping failure as
fatal; the low-level transaction's repaired-old publication is checked separately here.

Literal terminal marker (exactly once, successful exit required):

```
PASS: DS41_KV_BACKING_CARD ranks=4 startup_min=1 grow=1 prefix_pin=1 shrink=1 stable_va=1 bytes_exact=1 late_map_rollback=1 rollback_unmap_poison=1
```

Bind the complete command, checker SHA256, source commit and full coherent core/plugin cohort
SHA256 to a fresh guarded receipt. No receipt from another core or checker is reusable. This
tool is intentionally not registered with host ctest or installation.

## Initial model scope

Use native DS41, TP4 exact wire, deterministic, max sequences1, speculative tokens0 and
spec-ngram0; M16, P1/arena/DSpark/TBO/tier/streaming/LoRA/media off. The current live admission
checks all of these. Keep the default BF16 cache storage and packed-fast decoding off for the
first independent growth gate. Later packed combinations need their own fresh strict codec,
producer/fused/scalar/fast/WMMA gates from this exact ABI cohort.

Use both homes `home-sync`, all five plugins from one full rebuild, historical host-pool256,
GPU headroom256MiB, host headroom8192MiB, cache192GiB; repeat the functional pair at96GiB.
The first series compares the same unchanged original framed completions at the same explicit
maximum context, one fresh cold/default-backed reference versus each live arm. The upstream
sync merged base is the reply reference. Also compare the selected original requests against
their earlier merged-base receipt; an enlarged maximum must not silently create a new base.
No framing/chat-template change, modified prompt set or hidden warmup is permitted.

At512Ki and1Mi maximum context, run only the existing short requests, including the6817-token
Ulong8k request for actual growth and an original short request after release for reuse. This
proves large virtual geometry and short-request execution only. It does not prove a512K/1M
filled context, long-position RoPE or full-context prefill correctness or performance.

MAX_MIB must equal the actual common backend-rounded KV reservation, not a historical table
estimate or a cropped physical cap. The CLI refuses `--vram-kv-mib`; use the cold reference's
resolved logical pool budget and the new card receipt's granule, round upward physically,
then explicitly freeze MAX for its paired live arms. Prefix-on/off and packed-on/off may
change the layout, so they require their own bound. The source first funds that entire
maximum plus late allocations and max(startup,runtime) headroom before its initial commit.

MIN is a total floor, not extra backing above fixed state/window/dummy readers. Start with
MIN0/WARM0/HYSTERESIS0/HEADROOM256 (all suffixes explicitly provided); a second independent
arm uses MIN equal to one granule, WARM one granule, HYSTERESIS one granule and HEADROOM256.
Record actual bytes rather than assuming either arm starts at MIN. The second arm's spare
growth may be logged with cause=shrink because begin_plan performs policy reconciliation;
classify physical changes by actual versus old as well as retaining the cause field.

Tiny prompts cannot guarantee a growth event. Page0 readers pin the first granule in each
FULL layer stripe. For a2MiB granule, BF16 latent1024B rows cross after about2048 rows, but
packed288B rows can stay in the same granule through about7281 rows (alignment can change
the exact point). Thus an unchanged6817-token prompt is a concrete initial BF16 demand;
the eventual packed gate should use the existing roughly9.7K harness history. Require logs
showing actual positive growth on all ranks; no numerical threshold estimate substitutes.

## Lifetime and strict evidence

Run a fresh prefix-on cold/live pair using original Uharness turn prompts, official framing
and the original reply dependencies. Demand nonzero cached prompt tokens on reuse, identical
text bytes/prompt IDs/generated counts, and exact Kh group/rank/window/logit rows. Preserve
every request name/order. Use `session_id` only as an existing label, then release via the
supported `DELETE /sessions/<label>?keep_tokens=0`, or use supported loopback operator
`POST /reset_prefix_cache` and save its nonnegative entries_dropped. Confirm no session remains.
Schedule one original short request after release to enter the pre-admission shrink lease.
An empty /stats response alone does not prove physical shrink or prefix release.

The supported cancellation path is closing a streaming /v1/completions connection. Save the
numeric /stats sequence binding, observe at least16 generated tokens, close that connection,
wait until it is absent, then release any retained session prefix and run an original cold
short request again. A server-generated cmpl-ID is not the scheduler sequence ID. Compare
the cancelled response's observed prefix only; require exact next-request full text/counts/
Kh. Do not invent /cancel. This cancellation case is a separate lifetime receipt, not a
timed reply comparison (cancellation position can race).

Per live role, parse exactly one READY and four BUDGET rows. READY must match frozen explicit
options and its literal bounded cohort. BUDGET actual+deferred must equal MAX, actual<MAX,
fixed_experts1. Every transition epoch must have exactly ranks0..3, equal old/actual/logical/
reserved/granule/MIN/MAX/cause, whole-granule actual, old matching the prior published total,
and actual<=MAX. Require startup old0, actual<MAX, at least one subsequent increase and one
decrease on all ranks. Bind grow/shrink positions to workload phase boundaries; require the
prefix-held interval does not discard referenced hashes and release precedes the selected
shrink. Reject missing/duplicate rows, a READY-only run, engine refusal/fatal logs, forced
termination, API errors, changed cohort hashes or absent detailed Kh equality.

Price process startup, requests, release/cancel, drain, shutdown and restored production
separately; keep total lifecycle wall time. No extra expert-resident claim follows from lower
KV physical bytes. The current expert pool remains fixed and the full maximum stays funded.

## Remaining launcher integration

The source checker is complete; model orchestration is deliberately a prepared root integration
plan. The existing ab_window has no growth knobs, release/cancel phase receipt or exact maximum
capture, and must not be patched into a passing-looking generic runner. Root should add a small
guarded command which starts each fresh role, reads/resolves its actual KV budget, freezes the
five explicit options and lifecycle phases, runs original `ab_requests.py` receipt(s), performs
the supported release/cancel calls and retains all phase offsets. Then the strict validator can
bind Kh rows and physical transitions to phases. Host scheduler/mask/controller faults, current
full ABI-cohort build and fresh card gate must pass before that command enables model growth.

Always wrap the command with `work/gpu_cmd_window.sh`, verify both exact research locks and
production stopped before/after its work, and let the outer wrapper restore/verify QFN. The
wrapper logs the command's exit but does not propagate it, so save/check inner status separately
and require the final outer QFN completion/inspect evidence as well. No service operation
belongs inside the checker or client.
