GLM main TP4 cache: first integration slice (SOURCE ONLY)

This candidate composes actual Core route hooks, current-route NativeBackingOwner,
NativeInventory/Planner, the real Mover transfer fence, and Engine model/sampler
phase sequencing. It is a functional correctness boundary, not a serving build.
No author compiled or executed this candidate. Parent matching builds/checks are
required. Existing v4 Engine startup and all inherited HOST regression bodies are
preserved unchanged and are rebuilt against this exact new Core/private ABI.

Activation is CLOSED. Engine rejects the separate experimental glm5next_cache
plugin after plugin loading but before declaration/opaque kernel initialization,
Plan/Pools/model loading. No complete GlmMainTp4VerifiedSourceFactory exists in
this slice. The protocol harness cannot set source_admitted. One real packed
expert/rank and geometry metadata are not a complete model/source capability.

The new explicit profile is GlmMainTp4AqlFailStopV1. It requires actual GLM main
TP4/ranks0..3, 45 layers, routed layers3..44, 288 experts/top8, MXFP4/E8M0 four-
member local records3342336B, seq1/rows1..128, no draft/MTP, no XP/stager/tape,
no LoRA/deferred collectors, and AQL. The old PermanentV1 and HostOnly/TinyAqlV1
world1 guards remain distinct. Frozen ABI27 structs/version are byte-identical;
the optional new C capability exports are weak-probed by a separately built
experimental module. The standard glm5next module behavior is unchanged when
its new macro is not defined. RAD_BUILD_GLM_TP4_CACHE_EXPERIMENTAL defaults OFF;
its module is emitted only into research-glm-tp4-home, never normal install/home.

The exact passed CPU coordinator body is copied byte-for-byte. Actual Engine
wiring preserves Batch.n_ahead while disabling actual issue-ahead. Each current
route readback must agree across all four ranks before owner work, including
resident-only ranks. Actual Mover publication token/unit/region/ptr/event/nonzero
generation checks and a real transfer-stream completion precede all-rank READY.
WTAB invalidation is not used as a DMA fence. Frame activity remains set through
sampler and final collect. Stop admission serializes with begin; active-frame
errors use fixed nonthrowing process exit88 before application diagnostics,
owner drains, barrier shutdown, or destructors. This does not provide recovery
from a hung/broken GPU driver or permit partial-rank continuation.

Common generation must be one independently verified complete four-rank source
bundle. The Inventory uniform descriptor-generation getter alone does not prove
that capability. The sample packer's selected_raw_bundle generation differs by
expert/layer; it cannot serve as this whole-bundle identity. Rank-local layouts
and payload checksums remain separate. Full-source/budget/placement/ordinary-
weight loading, real four-rank payload correctness and inference are later gates.
Whole-layer all-expert misses are about0.9GiB/rank; this bounded first profile
makes no long-prefill or production throughput claim.

New manual HOST targets:
* glm53_tp4_cache_session_check --run-host-protocol
  Actual new Session/Builder/Ctx closed-admission methods:42 boundaries, truthful
  lookahead, frame-active sampler check, exactly-once collect, sticky coordinated
  failures, malformed INT_MIN layer, OOM/caller publication, idle-close rejection,
  and three exec-isolated exit88/no-unwind witnesses. Child exits have a10s local
  monotonic deadline, kill/reap containment, and nonblocking witness reads.
  The Engine gate check calls the actual private gate using an existing HOST test
  friend; it does not claim a complete Engine.start/TP4 runtime pass.
* glm53_tp4_mover_fence_check --radiance-home MATCHING_NEW_HOST_HOME
  Actual HostOnly source/Pools/Mover/upload/Residency/event/generation. The new
  mover_wait_fixture.h is a tracked test-only derivative with one raw_session
  parameter and one return after real bootstrap, before Owner/Ctx construction.
  It is never used to grant TP4 HOST capability. Real upload bytes are checked;
  semantic sync failure is injected AFTER successful actual transfer completion.
  Lease/ptr/event/generation remain held until explicit ordinary retirement. This
  is not a negative hardware-sync recovery test. Each check calls real Core APIs;
  no fake backend/private structure or source authority is introduced.

Matching whole-Core build, then fixture configure/build (parent only):
  cmake -S radiance-glm-main-tp4-cache-src -B BUILD_CORE [parent HOST options]
  cmake --build BUILD_CORE --target rad_core libref quant [parent regression targets]
  cmake -S glm53-main-tp4-cache-integration-prototype -B BUILD_FIXTURE \
    -DRADIANCE_SOURCE=ABS_NEW_CORE -DRADIANCE_HOST_BUILD=ABS_BUILD_CORE
  cmake --build BUILD_FIXTURE
The CMake wrapper verifies actual Core source-home and every current source pin.
For optional architecture compile coverage, enable the explicit experimental
option in BUILD_CORE and build glm5next_cache; do not load/activate its output.
Use the same compile/sanitizer mode and matching new headers/archive throughout.
Parent applies an outer process-group deadline and explicit HOST/GPU visibility.
No CTest/automatic execution/install/restart/serving target is added.

Re-run unchanged v4 manual Engine/Foundation/contract, compact normal/collision,
Native inventory, strict P3, unified23 and the existing nine Core regression
checks against the NEW whole Core. CMake cache/pin defaults were updated only in
this new wrapper. Original frozen prototypes/Core and source receipts remain
unchanged. verify_source_pins.py is read/hash only and checks whole snapshots,
all34 inherited bodies, the raw-fixture derivation, six peer-owned paths, and
reversible namespace-only source copies. It performs no target imports/builds.

Fixture-only v2 compile repair: parent matching whole HOST Core and explicit
experimental module built successfully. The v1 fixture compile failed because
session_checker.cpp called rad_dev_vmem_committed without namespace rad::.
V2 adds that namespace to both calls only. Same exact frozen Core and all
acceptance/geometry/lifetime/test bodies otherwise; no ASan/TSan suppression.
Preserved failure: build-glm53-main-tp4-cache-host-release-evidence-20261025T165408Z.
Parent reuses the matching new Core, and separately builds/runs this fixture.

Fixture-only v3 late-fault rendezvous repair: the first v2 runtime failed
'late source error after exact all-rank enter'. Coordinator had correctly
released the enter rendezvous, but rank3 could report READY(E_IO) before
other workers resumed enter and incremented their owner-work witness.
The new mutation3 test-only gate waits for all4 actual post-enter owner-work
arrivals before fault injection. It is bounded by10s and aborts immediately
on Coordinator failure, so unexpected enter refusal cannot strand a barrier.
One late-source witness and all4 arrivals are still required; no GU/success
assertion is weakened. No Core change. Failed source/runtime evidence retained:
validation-glm53-main-tp4-cache-release-v2-20261025T170513Z/tp4-session.log.

Fixture-only v4 source-review repair, before any v3 runtime: four() now records
watchdog firing and requires it remainfalse after all workers/timer join. A
watchdog rescue cannot count as successful intended error notification, even
if both errors use RAD_E_IO. Mutation3 also requires source_failure/rank3.
No Core or expected success/refusal behavior changed; strict checks strengthened.
V3 remains frozen/unexecuted. All parent runtime/sanitizer checks are pending.
