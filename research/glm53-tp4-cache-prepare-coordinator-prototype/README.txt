Closed GLM TP4 cache prepare coordinator: HOST executable specification

NEW standalone prototype. No Core links/edits, actual Engine, model, Native
inventory/owner, GPU, collective, queue or launch. It cannot enable TP4 cache
serving. Parent alone compiles/runs Release/ASan/TSan. Existing TP1 guards and
frozen earlier prototypes remain unchanged. No author compiler/runtime/device.

This is a CACHE-SPECIFIC PROCESS-FAIL-STOP first profile, NOT graceful arbitrary
collective/queue fault recovery. It supersedes the earlier design's general
collective-transaction prerequisite FOR THIS NARROW process-fatal experiment.
Successful gates authorize local GU work only; no collective transaction or
exactly-once peer packet API is implemented here.

Closed profile and integration meaning
- Exactly4 ranks0..3, actual GLM canonical experts288/topk8, max_rows128, one
  sequence, prefill1..128 or true decode1, layers3..44 in original order42 gates.
  One whole-step slice first_row0 per layer, as stock c.rows=g.max_tok currently
  does. Batch256/chunked/multi-sequence/MTP/draft/spec/ahead/lane1/replay/XP/heat/
  stager/alternate expert geometry are NOT admitted. Parent must not expand a
  current32record toy inventory merely by changing world_size.
- Pass identity contains immutable model/common source generation, uint64 epoch,
  actual step/phase/draft/spec/ahead, token/sequence/output counts. Common source
  generation means verified logical bundle revision, NOT rank-local checksum.
  Rank payloads/descriptor checksums differ under TP and are checked locally.
  A future Core adapter translates actual RAD_PHASE constants to this explicit
  portable enum; it must not assume their integer values match.
- enter compares exact complete canonicalIDs[token,topslot], including-1 drops,
  after actual routing event and local coverage validation. All4 must enter and
  agree BEFORE local cache owner work. It does not inspect real route buffers or
  copy payloads. Counts/offsets/sorted validation remains the actual Ctx's job.
- Local cache acquire/validation/upload-ready setup occurs OUTSIDE the mutex.
  ready requires all4 successful local statuses BEFORE any GU grant. No owned
  cache miss is needed on a rank; all-resident/no-work ranks still participate.
  The actual adapter must prove its stated event readiness; status0 itself is
  not GPU completion, checksum validation or a lifetime lease.
- Fast ranks cannot overwrite a gate until all peers consume its GO. Epochs and
 42 ordinals are checked, normal early rank return fails, and source/metadata/
  explicit close errors wake waiters with the same first sticky POD failure.
  finish and begin cannot silently restart after failure or wrap an epoch.

Fatal policy, rather than generic GPU recovery
Coordinator methods return negative statuses so HOST tests can inspect votes.
THE ACTUAL FIRST EXPERIMENTAL PROFILE MUST call
  require_success_or_process_stop(status)
immediately on a negative cache/coordinator/early-rank status. Its fixed action
is std::_Exit(88), before stream waits, C++ teardown/destructor unwinding,
request publication or further model work. No arbitrary user-supplied handler
or Ready/source callback can turn this into a recoverable success path.
Engine rank guards and Ctx failure paths must notify then take that fixed stop
action before lengthy diagnostics/end_pass/drain, even if a rank never reached
enter. Normal request cancellation must not make one rank skip an active gate;
initially defer it to the shared pass boundary. Shutdown/known errors are fatal
for an active experimental pass. Already-written private outputs/state are not
rolled back, and surviving requests are not resumed on that Engine.

This avoids requiring a generic prepared-collective protocol merely to TEST
cache failure containment: a known cache error terminates the experimental
process rather than gracefully waiting on possibly unmatched peer work. It
does NOT establish GPU quiescence, safe memory free, driver recovery, bounded
process death on a wedged driver, or recovery from a hardware hang which never
returns a host error. External idle-window/health supervision remains necessary.
The subprocess test proves CPU _Exit/no-destructor semantics only. It is not a
claim about OS/driver device cleanup or production serving availability.

Future narrow Core wiring (not done)
Engine owns coordinator before rank release, opens common epoch, and installs
an early rank guard before mine/bind/run_step. Missing batch, thrown allocation,
negative issue/prepare and short successful return notify/fail-stop. Ctx's actual
rad_expert_prepare hook gains enter before owner.prepare and ready after checked
local readiness; only GO returns to GU. Existing raw source/typed inventory and
real row/copy/lease validators remain mandatory. Existing TP1 guards stay until
a separately versioned explicit fail-stop profile, actual full rank geometry,
source ownership and fatal policy have been wired/tested. PermanentV1 must not
silently change meaning. No new capability is advertised by this prototype.
Original stock GLM route_prepare is currentlyoff and must be explicitly enabled
only in that requested cache profile. A future graceful Engine mode may still
need stronger collective prepare/commit; it is not the first-profile prerequisite.

Bounds and lifetime
Coordinator uses fixed arrays for4*128*8 IDs=16384B plus fixed POD/mutex/CV and
rank counters. static_assert(sizeof<24576) protects a24KiB logical object quote;
no hot dynamic allocation is requested by its state machine. Mutex/CV runtime
internals and general allocator/RSS are not claimed as a physical quota. Four
test threads have their normal platform stacks (commonly4*8MiB); this is a
standalone HOST tool, not Engine budget admission. Callers must join threads
before destroying Coordinator; concurrent destruction is not supported.
No gate abort authorizes freeing a real source lease/landing/arena. The actual
process-fatal adapter deliberately avoids that cleanup inference.

Manual parent validation
  cmake -S . -B ABS_NEW_BUILD -DCMAKE_BUILD_TYPE=Release
  cmake --build ABS_NEW_BUILD --target tp4_cache_prepare_check
  ABS_NEW_BUILD/tp4_cache_prepare_check
No CTest/automatic execution/install/Core build. Compiler/sanitizer flags remain
parent-owned. Author hash/source checks alone are not a test PASS.

Authored matrix:9 cases,24 refusal groups,336 local-GU counters in two fully
successful42layer passes. This does not count all extra edge-case GO operations
as executed kernels; every GU here is an atomic HOST counter.
1 full128row prefill then1row decode; all-resident rank still votes every layer.
2 REAL POSIX short read on rank3 after other3 reach ready, allGU denied.
3 rank3 normal early-exit notifier without entering; three CPU waiters wake.
4 exact canonicalID mismatch before any owner callback.
5 eleven pass/key/ID/rank mutations and six unsupported initial profile variants.
6 prior GO work remains; later known failure denies subsequent GO, no rollback.
7 close wakes/denies restart;8 last valid epoch then overflow refusal.
9 subprocess with4 workers: known negative exits88 without C++ destructor marker.
Controlled scheduling observes ready/entered masks under the real mutex with a
bounded10s wait; expiration first aborts the gate, so tests do not intentionally
leave absent-rank deadlocks. No deadline is treated as a successful vote.
