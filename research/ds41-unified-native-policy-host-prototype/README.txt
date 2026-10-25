Unified Native preparation / permanent failure policy HOST regression

SOURCE ONLY. Parent owns matching whole-Core build and all execution/sanitizers.
No child compiler, numerical/model or device execution. Original B2a/ready,
policy, frozen Radiance and oracle folders remain untouched.

Core candidate is the separate radiance-unified-native-policy-src composition.
Its four source resolutions are recorded outside the tree in
unified-native-policy-resolution/RESOLUTION_MANIFEST.json and parent diffs.
The public ABI27 structs are unchanged; private Ctx headers MUST match the whole
Core archive. This CMake checks the build's CMAKE_HOME_DIRECTORY accordingly.

This checker is a NEW derivative of the frozen Native plan/owner checker.
All14 original test group bodies remain, and Fixture requests immutable Program
PermanentV1 BEFORE declaration. Actual libref HOST MoeFP8 rows, real route-copy
bank, S1/cache NativeRecordSource, NativeBackingOwner, Mover, Pools and Residency
are still used. No launch/collector instance is replaced. Existing test-only
copy/event failure wrappers are retained.

Nine additional groups (23 total planned):
- strict unsupported Native n_ahead begin: no architecture/prepare/copy/source
  work and permanent first failure;
- direct Native prepare outside active pass: permanent refusal before copies;
- actual source refusal E_NOMEM followed by ONE rank-thread allocation failure
  at the diagnostic string construction: helper catch must preserve the first
  code with its bounded allocation-free fallback diagnostic;
- prior step failure before Native prepare, with and without Native owner:
  direct prepare must preserve first error, including before no-owner fast path;
- Legacy unsupported Native begin remains fatal for that Native owner while
  Ctx terminal_status stays OK (no silent default-policy change);
- Legacy first rad_step_fail(A), later forbidden rad_lane(1) statusB, actual
  owner end_pass and retry all preserve A. Only Native's sticky first status is
  corrected; default Legacy programs without Native preparation are unchanged;
- strict AND Legacy collector+Native-Owner admission are refused by the actual
  existing expert_prepare_open guard. A counted test-only DEVICE metadata row
  proves neither body nor collector is launched; source reads/Owner prepares/
  Residency publication stay unchanged. No device instruction executes.

On each strict failure, actual run_step/direct prepare/set_lane/weight_ptr
must return the same first error before architecture entry, route copies,
readiness waits, payload allocation or source/publication changes. First owned
message persists. Actual shutdown/retirement is allowed; a later init on the
same Ctx must refuse before payload allocation or rebinding.

Native/deferred frame ordering
  Private two-phase admission checks terminal/checkpoint/spec and Native fatal/
  profile status BEFORE deferred_status_.begin can reset/enqueue a frame.
  Successful pure Native admission invokes no owner callback and does not mark
  active. Only after successful deferred begin does nonfallible activation mark
  active; all subsequent pass exits retain owner end_pass cleanup. Native end
  remains permitted after terminal failure for retirement.

  Currently Ctx::expert_prepare_open rejects checks_device_status() + owner.
  Therefore a reachable collecting Native-owner pass is NOT fabricated here.
  Combined strict/Legacy tests establish this admission refusal; the exact
  two-phase source order is reviewed, not claimed as a live combined GPU test.
  Existing queued-collector/checkpoint tests and legacy Native checker must also
  run against the newly compiled composition. No public ABI/version is added by
  this merge; existing Core+Engine capability1 is preserved from validated policy.

Failure scope and bounds
  Actual Native source/cache/landing/readback caps and all14 baseline geometry/
  guard tests are inherited; no cap is increased or full model data used.
  New metadata collector has zero operands/scratch and is never launched.
  One additional allocation fault is thread-local and test-only; it deliberately
  targets std::string diagnostic construction after a source refusal, never a
  kernel body or queued device transport. It automatically disarms on one throw.
  Throwing/nothrow scalar/array/aligned new and matching unsized/sized/nothrow
  delete are all replaced consistently with malloc/posix_memalign + free.
  A direct allocation-family contract covers normal/aligned/nothrow paths and
  the stable_sort nothrow-new -> sized-delete pair without disabling sanitizer
  mismatch detection. The model diagnostic OOM witness remains unchanged.

  Direct lower-level Mover retry after explicit drain remains its existing
  trusted protocol; this is distinct from permanently failed model Ctx reuse.
  Native readback quarantine on failed drain is retained, but broader Arena/
  Rank teardown after hardware/queue failure remains an open lifetime gate.
  No safe GPU fault recovery, rollback, TP, Engine Native activation or serving
  integration is asserted. Same-thread Ctx/owner lifetime rules are unchanged.

Parent-only build/run
  Build radiance-unified-native-policy-src as HOST first. Then configure HERE
  with RADIANCE_SOURCE and matching RADIANCE_HOST_BUILD. Target:
    ds41_unified_native_policy_host_test
  CMake also builds admission-only unified_policy_collector MODULE locally;
  its absolute target path is compiled into this checker, never installed.
  Run manually:
    CHECKER --radiance-home MATCHING_UNIFIED_HOST_HOME
  No automatic CTest/build/install/device execution occurs from this source.
