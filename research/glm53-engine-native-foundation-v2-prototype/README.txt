GLM53 tiny Engine foundation — Patch A v2 corrective revision, source only

This NEW Core/package implements the approved transport foundation. No Engine
request, Engine::start branch, scheduler, model step, serving CLI, deployment or
restart is activated. The author has not compiled or executed these sources.
Every new result is pending matching parent builds/runtime and independent review.

V2 corrections, with the original1492/45 parent frozen
  F1: The real preserved MoeFP8 route declaration has classes2. The pure Ctx
  quote now admits the reviewed route classes1/2 and enforces actual nonnegative
  class_index==slot_index%classes, while retaining every existing bounds/member/
  source guard. Actual two-class declaration remains unchanged. Additional
  one-class coverage mutates only bounded route metadata for API quote testing;
  it is explicitly NOT a genuine re-declaration, numerical success or a Native
  cache-layout class claim. Seven class/modulo refusal cases preserve output.

  F2: admit() now debits global_control_allowance into an explicit admission-owned
  Controls baseline BEFORE admitted_ becomes visible or any generic ticket can
  exist. It has no externally resettable ticket. Generic/stream control tickets
  can use only the remaining category; release cannot lower the baseline. The
  Budget ends only after every ticket/stream/worker retires, then its logical
  admission credit ends. This does not assert destruction of backend-global
  retained caches. New cases reject spending the entire Controls cap again,
  admit exactly its remainder, reject one more byte, and verify the reservation
  remains after generic and stream-ticket reset. Whole cap/category sum unchanged.

  Confirmed original parent fixture compilation error RAD_DT_BF16 is corrected
  to actual RAD_BF16 in the NEW manual checker only. Original whole HOST Release
  Core built, but its fixture compilation FAILED; ASan/TSan did not start.
  Original failure log and SOURCE_RECEIPT are preserved/pinned. No runtime PASS
  is inferred from that build, and no author compiled or executed v2.

Core ownership and source changes
  Base: frozen radiance-glm-tiny-engine-native-src (1487 files). New isolated Core:
  radiance-glm-engine-native-foundation-v2-src. The full-tree map and minimal diff
  name every addition/change. Private Backend/Program/Plan/Ctx headers require a
  matching whole-Core rebuild. No old private archive is mixed with new headers.
  Existing Core CMake CONFIGURE_DEPENDS glob includes both new build/*.cpp files;
  no implicit runtime/Engine target or Core CMake behavior is introduced.

  tiny_startup.{h,cpp}: same14 category caps,135244800B reservation inside the
  unchanged201326592B cap. Move-only tickets are acquired before the manual path's
  payload/control requests; no zero-size dangling ticket is admitted. Maximum
  three streams and two auxiliary workers. Actual HOST backend threads plus
  workers total at most five; checked pthread defaults are <=8MiB/4096 guard.
  HOST category uses actual Task ring + private stream size + explicit control
  allowance + checked stack/guard. AQL category preserves the requested three
  queue/argument/bounce stream terms38538240B; private structures/control use the
  separate4MiB control reservation. Backend globals, signal/audit caches and TLS
  have actual private size facts exposed and an explicit control allowance.
  Category reservations and requested-allocation tickets are NOT process RSS or
  a process-wide allocator interposition quota. Opaque vendor/driver allocation
  overhead, allocator arenas, stdlib internals and external concurrent activity
  are explicitly outside such a claim. Stable pthread defaults are required;
  changed queried defaults refuse. No unknown backend silently receives a quote:
  appended optional private quote/creation hooks missing => UNSUPPORTED.

  HOST construction owns stream/ring through resize/thread exceptions and writes
  caller output only after successful worker creation. AQL construction likewise
  owns every actual returned queue/karg/bounce handle through allocation/access
  and vector-growth failures. AQL tiny creation verifies actual returned queue
  depth BEFORE argument/bounce allocation. This improves partial cleanup without
  changing successful normal startup. The actual device allocator remains in use.

  Pure Sampler/Ctx/NativeBackingOwner/StepSlot providers quote actual private
  types and allocation loops. See the separately frozen component handoff and
  source pins for exact field/count semantics, unsupported profiles and explicit
  growth/control allowances. No fake private structs or W WeightInfo snapshot.
  NativeBackingOwner success still requires the actual NativeBacked Plan and
  exact inventory capability; a metadata-only Plan is explicitly refused.

Actual tiny RelayoutPool seam
  core/build/tiny_loader.h exposes a manual research function, not Engine/CLI
  activation. It calls the actual existing RelayoutPool/run/relayout code with
  immutable fixed limits:2workers,1MiB input arena,two128KiB pinned slots per
  worker, output/input per job <=128KiB,max59 jobs and aggregate input pages
  <=1MiB, overall loader quote <=2MiB. Destinations, complete selected-plane
  spans, row products, actual kernel/layout references, duplicate/overlapping
  spans and source indices are checked before allocation. The tiny branch has
  fixed admitted input offsets and job ring; generic input fallback, hole-node
  allocation and arbitrary std::function captures are absent on that branch.
  Fill contexts/Program/destinations are borrowed fixed manual inputs for the
  call and must stay immutable/alive through return. This seam does not confer
  a GLM source seal or accept a production/arbitrary-source factory.

  Both WorkerState/Bounce owners are Pool members allocated before workers.
  Slots become busy BEFORE memcpy; failed copy/record may still have enqueued.
  Every refusal stops future shared submissions, retains the possibly enqueued
  slot, joins ALL started workers, and performs a REAL shared-uplink drain while
  both Bounce owners remain alive. Only a successful real drain permits event,
  stream, pinned/input release. Negative/throwing real drain exits86 on this
  explicit isolated research path before release. Ordinary default branch still
  uses its existing8x4MiB slots/256MiB retention and normal Engine callers.
  Destination rollback is not promised on terminal load failure. Failure receipt
  is diagnostic only, never Ready/result authority. Before-Pool OOM has no work
  and all_retired=false; constructed-Pool cleanup records actual retirement.

New manual cases (intended assertions; no author runtime PASS)
  Actual private HOST/Task/pthread/Sampler/StepSlot quotes and caps, output
  preservation; all three HOST constructor new-failure positions; actual loader
  destination bytes; five explicit research semantic fault phases; two actual
  pinned allocator failures; two early C++ staging/control OOM positions; real
  copy/record/wait wrappers that first perform the actual operation successfully
  then report failure; proven final real drain and two retained Bounce owners;
  caller quota refusal before allocations/publication. Requested C++ controls
  are witnessed against the named allowances, not called allocator RSS.
  A subprocess closes a genuine HOST gate, observes actual stream_sync STATE,
  then demands exit86 before any release. This is distinct from semantic result
  injection after successful operations. It does not pretend to recover a GPU.

  contract/ contains the frozen successful four-artifact219tensor/real320-token
  checkpoint, real architecture declaration and actual same-object source/cache
  cases. Existing7groups/24shape/7graph/11SamplerOOM/7source/10tokenizer bodies
  remain, with additive pure Sampler/Ctx quote checks against the finished graph.
  No Ctx, Owner, Pools, Engine, source payload transport or model step is run by
  this checker. The actual source/package remains EngineEligible=false.

  regressions/compact keeps byte-identical normal/collision/inventory/strict-P3
  source bodies. Original23group unified checker remains referenced from its
  frozen source. regressions/device contains byte-identical LATEST persistent-
  input DEVICE checker/helpers, including actual same-stream snapshots and no
  extra synchronization. Only CMake roots/build defaults and Core pin JSONs are
  adjusted. Numerical DEVICE execution remains a separate parent/idle gate.

Parent-only commands (run compilers serialized; never run by author)
  python3 HERE/verify_source_pins.py
  cmake -S CORE -B NEW_HOST -DRAD_WITH_HIP=OFF -DRAD_WITH_FFMPEG=OFF \
    -DRAD_BUILD_TESTS=ON -DRAD_BUILD_DS41_EXPERIMENTAL=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build NEW_HOST --parallel 1
  cmake -S HERE -B NEW_CHECK -DRADIANCE_SOURCE=CORE -DRADIANCE_HOST_BUILD=NEW_HOST
  cmake --build NEW_CHECK --parallel 1
  NEW_CHECK/tiny_native_foundation_check
  NEW_CHECK/contract/tiny_startup_metadata_check --run-host \
    --radiance-home ABS_NEW_HOST --architecture ABS_NEW_CHECK/research-architectures/libglm53_tiny_startup_arch.so
  NEW_CHECK/regressions/compact/compact_authority_test
  NEW_CHECK/regressions/compact/compact_authority_collision_test
  NEW_CHECK/regressions/compact/compact_native_inventory_test --radiance-home ABS_NEW_HOST
  NEW_CHECK/regressions/compact/strict_p3/glm53_native_source_strict_policy_test --radiance-home ABS_NEW_HOST
  NEW_CHECK/regressions/compact/frozen-unified-regressions/ds41_unified_native_policy_host_test --radiance-home ABS_NEW_HOST

  Repeat with matching whole HOST ASanUBSan and TSan Core/archive dependencies,
  matching checker flags and original parent sanitizer launcher policy. Source
  author does not fabricate or relabel any of these as passing.

  After the separate matching HIP/AQL Core is built, DEVICE compilation only:
  cmake -S HERE/regressions/device -B NEW_DEVICE_CHECK \
    -DRADIANCE_SOURCE=CORE -DRADIANCE_BUILD=NEW_AQL
  cmake --build NEW_DEVICE_CHECK --parallel 1
  Actual --run-gpu execution/default+host-kargs require the parent's fresh idle
  gate and matching frozen row module; no configure/build auto-executes it.

Patch B remains gated on parent compilation/runtime/peer review of A. Patch B
will use the SAME singleton Engine-owned Checkpoint/cache, actual S1/P3 source
factory, actual whole-ledger postdeclare quote, ordinary-loader completion,
persistent Native owner and real Rank Ctx lifecycle. This A is not that claim.
