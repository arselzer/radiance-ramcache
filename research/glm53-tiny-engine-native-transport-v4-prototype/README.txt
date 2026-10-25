Actual Engine.start native-cache transport v4 SENTINEL FIX — SOURCE CANDIDATE

This NEW isolated Core composes the passed Foundation v2 Core/v3 checker,
source-factory v2 and actual Engine methods. Source authors have not compiled or
executed this candidate. Parent matching compilation/runtime and independent
review are pending. Serving, model/sampler steps, GPU/full GLM/TP4/MTP execution
and deployment/restart are absent. No public production files are changed.

Core: radiance-glm-engine-native-transport-v4-src. Existing frozen parents remain
intact. Current factory and all code paths are closed to the same tiny synthetic
HOST profile; this is an actual Engine integration proof, not an almost-enabled
full-model cache. Real GLM stock architecture support is already present and a
stock max_spec0 fit test may be useful independently of this research path.

Actual startup path
- Fixed internal TinyNativeHarness::create admits known categories before real
  Engine construction, source/registry/sampler/thread/stream/pool work. Options
  are only bounded absolute checkpoint/home/work paths and an explicit manual
  phase fault. It accepts no caller catalog/provider/Ready callback. All RADIANCE_*
  environment overrides are refused for this fixed fixture. A private staged
  home contains exactly libref, libquant and the unchanged tiny architecture.
- Engine::start dispatches its tiny-only branch; ordinary/default start remains.
  The same Engine read_metadata/load_vocab/load_plugins/declare/declare_sampler,
  load_weights/configure_kv/step_state_init and actual Rank Ctx are used.
  The checkpoint's zero heads pass only the sealed same-object profile guard.
- Engine owns one source binding/Checkpoint and one actual resolver cache.
  All source, importer/refusal, loader and vocabulary consumers use central
  delegates. Source_fn strongly retains the SAME binding. No second checkpoint
  open/cache or full WeightInfo snapshot is created.
- Declaration runs once in one precharged/joined auxiliary worker, catches errors
  and uses actual rad_tools_declare(instances=false) then MOVES its single Builder
  Program into final Rank storage. Real Engine sampler adds PERSIST sampler.tokens
  I32[1]; exact finished buffer/arena counts are measured by actual quotes. Shared
  initialize_program_instances reuses the real Builder loop after admission;
  the closed quote permits no opaque init callbacks or existing instances.
- Distinct stable metadata Plan lives in TinyState for the source factory.
  Actual same-checkpoint219range/192piece physical→S1capture32→3shard→Reader→P3
  source construction yields24owned/8resident units. Final Rank::plan is separately
  NativeBacked BEFORE configure_native. The source identity/lease/work ownership
  remains the frozen factory algorithm, not a substitute catalog.
- Real pooled backing allocator delegates rad_device_allocator and retains its
  own accounting owner beyond Pools. HOST, like AQL, has elastic VMem: two normal
  backing requests and separate rounded unused KV VA/zero commitment are billed
  inside unchanged6MiB. No malloc allocator stand-in or duplicate original owned
  payload backing is introduced.
- Compute is an actual Rank TinyStream before Mover/owner/Ctx. Mover's actual
  TinyStream is precharged before bootstrap, and its private retirement hook
  drains both streams before release. CtxDesc's early tiny retirement policy
  covers internal init failure shutdown before oracle/xp/WTAB/arena frees.
- Actual load_weights performs real source encoding and weight_share before
  typed omission. It omits96owned views/626688 source bytes BEFORE destination
  lookup/WeightInfo.ptr publication/staging. Exactly59ordinary views/393856 bytes
  use the SAME actual RelayoutPool/run/relayout implementation under a2MiB ticket.
  Fixed named checkpoint fill contexts replace captured lambdas only in this
  branch. Complete TP1 rectangles preclude generic64MiB partial-column fallback.
- After both worker joins and final REAL shared-uplink retirement, independent
  ckpt_plane_rows comparisons check59actual ordinary destinations. Actual S1
  bootstrap compares32member spans of8resident records, including encoding/unit
  authority, without comparing uninitialized inter-member/end record padding.
  Only then does finish_bootstrap trim the actual cache before persistent
  NativeBackingOwner::open and actual Ctx::init.
- Actual loaded Vocab BOS257/EOS256/EOG{256,258}, TokenView and grammar index are
  checked; index bytes are reported/capped inside metadata. configure_kv keeps
  the real no-KV sentinel. Actual step state is initialized with a retained
  TinyState EOG vector, which survives a copy that enqueues then reports failure.
- Only the last nonfallible publication marks TransportReady. Engine::serve
  refuses this mode. Same Engine cannot retry any startup attempt: the earliest
  error is sticky even before Ctx. Explicit retirement is idempotent. A retired
  Engine refuses start even if it had never attempted startup; no released
  admission credits can be reused.

Lifetime/budget distinctions
Same14 category caps/192MiB conditional requested-owner cap; no increases.
Runtime and arena admission cover actual component quotes; real backend event
objects have separate precharged control credits plus admission-owned global
baseline. Engine/source metadata and the small fixed bootstrap control are
charged, and known private sizeof/actual count results must come from parent
execution. This remains neither RSS nor a process-wide allocator interposer.
Vendor/driver/allocator/stdlib-global overhead, OS page cache and external users
are not proved by it. Embedded object bytes can be conservatively double-billed
inside category upper bounds; they are not fabricated exact malloc observations.

On every partial/normal close: retain sources → drain actual Mover/compute →
owner quiesce → guarded Ctx shutdown → owner close → actual step/EOG free →
source close while Program/metadata Plan still live → ranks/owner/Mover/compute/
Pools destruction → source generation/cache/Checkpoint → ticket release. The
ledger precedes ranks in Engine and outlives all tickets/callbacks. A genuine
negative or throwing drain/close fails stop before unsafe dependent release.
No GPU recovery is claimed. The factory's known simultaneous-FD-exhaustion
staging-directory cleanup limitation is retained, not hidden as general recovery.

Mechanical source composition
FACTORY_COPY_NAMESPACE_MAP.json pins every frozen factory/helper source and its
Core copy. ONLY five C++ namespace prefixes change in those Core copies; reversing
that map byte-recovers every body, including predicates/caps. This prevents
whole-archive duplicate symbols when unchanged original contract/P3 regressions
compile too. The factory external API remains glm53::engine_source. Frozen
source-factory v2/Foundations and original tests are not edited or relabelled.

Manual cases (assertions authored, execution PENDING)
Actual Engine checker has6groups: complete startup+Ready+no-serve+explicit close;
create→retire→start refusal with calibrated complete C++ allocation-family and
actual source open/pread, device allocation/upload witnesses;
9intended phase refusals with same-Engine sticky retry and cleanup checks;
3actual device allocator refusals (Sampler/pool/loader); and an actual successful
8-byte EOG submission whose wrapper returns failure before the normal sync.
An added sixth pure-quote group declares the actual tiny graph independently
and checks18 nonzero/real KV mutations preserve refused output.
The EOG case requires the intended ContextBound stage, real retirement, no retry
and no leaked backing/generation. All wrappers delegate actual backend calls;
no fake Backend/private structs or numerical Model step is introduced.

All previous Foundation v3, seven-group source/declaration contract, compact
normal/collision/inventory, strict P3, unified23 and persistent DEVICE source
bodies remain identical; only matching Core pins/build paths differ. No GPU
checker auto-runs. The parent builds every private source with matching headers.

Parent-only sequence (not run by source author)
  python3 HERE/verify_source_pins.py
  cmake -S CORE -B NEW_CORE -DRAD_WITH_HIP=OFF -DRAD_WITH_FFMPEG=OFF \
    -DRAD_BUILD_TESTS=ON -DRAD_BUILD_DS41_EXPERIMENTAL=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build NEW_CORE --parallel 1
  cmake -S HERE -B NEW_CHECK -DRADIANCE_SOURCE=CORE -DRADIANCE_HOST_BUILD=NEW_CORE
  cmake --build NEW_CHECK --parallel 1
  NEW_CHECK/tiny_engine_native_transport_check --run-host \
    --radiance-home ABS_NEW_CORE/radiance_home \
    --architecture ABS_NEW_CHECK/research-architectures/libglm53_tiny_startup_arch.so
  NEW_CHECK/tiny_native_foundation_check
  NEW_CHECK/contract/tiny_startup_metadata_check --run-host \
    --radiance-home ABS_NEW_CORE/radiance_home \
    --architecture ABS_NEW_CHECK/research-architectures/libglm53_tiny_startup_arch.so
Inherited compact/P3/unified targets are under NEW_CHECK/regressions/compact.
Repeat matching whole Core/helper ASanUBSan/TSan under the parent's serialization,
resource and sanitizer launcher policy. There is no CTest/implicit execution,
network/download, installation, production config or service target here.

Shortest next real-GLM path, separately from this integration checkpoint
Use actual stock GLM metadata/kernel layouts and TP4 shards, not bigger synthetic
constants. Cache selected experts after real routing, preserve last-reader
ownership, coordinate rank-local source failures BEFORE collectives, and quote
actual shared process RAM rather than applying this tiny192MiB ledger. Check
whether selected r4d weights require relayout; current canonical identity-only
S1/native acceptance cannot simply be widened numerically. Existing file-tier
read caching is a possible quick experiment but still reads whole routed layers
before routing and direct checkpoint/TP-shard restrictions apply. No extra tiny
framework feature is planned as a prerequisite to a stock max_spec0 full-model fit.

Revision v2 / narrow lifecycle repair
The original v1 frozen Core/prototype remain unchanged. Independent review found
create→retire before the first start released runtime tickets while attempted
remained false; v1 could then start and its retired cleanup would be skipped.
The only v2 Core difference is an early retired/retiring/Terminal/Retired guard,
before setting attempted or invoking any startup method. Existing earliest-error
behavior is unchanged. The new fixture group requires zero new C++ allocations
(complete inherited aligned/nothrow/scalar/array family), zero actual device
allocations/uploads and zero calls to the concrete source open/pread seams around
the refused call; source-I/O and heap witnesses are independently calibrated.
No allocator hook claims process-wide malloc/RSS interception. The four original
semantic group bodies remain unchanged, apart from the copy observer common to
the new no-work witness. Parent execution is required; no PASS is claimed here.

Revision v3 / diagnostic ONLY after actual v2 refusal
Parent whole matching v2 Release compilation PASSED, but the FIRST actual Engine
startup returned-18 after configure_kv/prefix-disabled, before Ready. Frozen
validation-glm53-engine-native-transport-v2-release-20261025T154557Z/receipt.json
and engine-transport.log preserve this real failure. No runtime PASS is claimed.
This new revision records actual post-bind KV group count, sentinel base-present,
bytes/stride/rank/layer slots/dtype and second Ctx quote/init statuses; the manual
baseline prints those bounded fields before its existing failure assertion.
The source hypothesis is a legitimate null group0 binding whose base aliases
the reserved KV VA while bytes/stride/rank/layers are zero. V3 DOES NOT normalize
that binding, adjust the quote predicate, change data, add a permissive KV case
or increase any category. Parent must confirm actual fields before a separate
corrective revision is considered. Existing v2 terminal guard/tests are retained.

Revision v4 / exact inert group0 correction, v3 diagnostic preserved
Frozen v2 actual failure and frozen unexecuted v3 diagnostics remain unchanged.
Source tracing confirms KVManager inserts its null handle0 plan, Pool::reserve(0)
returns the reserved base, and bind_kv_pools legitimately carries that pointer
with zero bytes/stride/rank/shape/layers. The matching quote now requires exactly
one genuinely inert Program group0 (every declaration/size/layer/flag field zero)
and optionally its fully zero-extent binding0. Only the inert base is ignored,
without dereferencing it. No real KV group, bound dtype/shape, extra group or
nonzero extent is admitted. Empty preconfigure bindings still require that same
true Program sentinel. Core runtime/ctx.cpp is the independently authored narrow
repack predicate; all other Ctx/Mover lifecycle behavior remains unchanged.

The actual Engine baseline demands observed non-null sentinel base plus zero
bytes/stride/rank/dtype/layer-map/all shape extents and successful quote/init. The
separate sixth group builds the same ACTUAL tiny source/architecture Program
without sampler allocation or instances; a clearly marked opaque stack marker
exercises only the pure quote API, never a backing/Ready claim.18 mutations cover
bytes/stride/rank/shape0/shapetail/layer map/dtype/extra binding/extra Program KV,
heads/layers/block size/state bytes/declaration dtype/name/zero/replicated/drafter.
All earlier33 inherited fixture bodies remain byte-identical. No Core numeric
category/limit, source or tensor data, Loader behavior, model/sampler execution
or serving path changes. Parent matching whole builds/runtime are PENDING.
