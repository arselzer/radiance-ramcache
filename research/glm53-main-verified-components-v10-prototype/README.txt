V10 actual retained metadata count consistency repair

Preserves frozen V9 and actual failed native-metadata diagnostic. The V9 helper
invented a4096 KV leaf ceiling; the pinned real config alone flattens to20454
leaves, so it rejected legitimate metadata before any native diagnostic/source
scan. V10 removes that guessed ceiling with NO replacement guessed model bound.
It preserves exact vector-length, retained string/pointer, Program alias and
post-return actual profile checks. m.n_kv must be nonnegative, then its safe
size_t conversion must equal the owned key count. Vector counts, metadata count
and pointer-array identity now have separate observed-count failure diagnostics.
The JSON records actual n_kv/key/value/pointer counts in
retained_declaration_metadata. All source/Program/Core/admission/layout/math and
existing refusal bodies remain unchanged; this adds no authority or readiness.
Build/runtime validation remains pending parent; author source/hash checks only.

V9 compile-only diagnostic type correction

Preserves frozen V8 and its actual compiler failure. The ONLY C++ change converts
MoveUnit::reason (const char*) to a null-safe std::string using a strnlen bound of160. This preserves the
original diagnostic output limit without unbounded strlen-before-substr. All retained MetaOwn ownership, post-return
metadata guards, Native diagnostics, refusal bodies, source/Core and CLI/schema
contracts are unchanged. Same exact CoreV5; no whole-Core rebuild needed. Parent
compilation/runtime validation remains pending; author performed source/hash only.

V8 retained declaration-metadata lifetime repair and pre-I/O native diagnostics

SOURCE ONLY. Parent complete prewalk V7 failed status-18 at real NativeInventory
bind after684.85s; QFN was never stopped and no model kernels ran. The source
contains an independently confirmed fixture lifetime defect: MetaOwn was local
in declare_actual, finish() published strings/KV pointers into that local, and
Builder copied only RadModelMeta raw pointers into all4 Programs. The local died
on return, before the complete Reader/native admission. Earlier in-helper checks
therefore did not establish later metadata validity. The failed V7 source and
receipt remain immutable; attribution of its exact runtime failure still awaits
parent rerun, since its diagnostic message was also constructed before bind.

V8 keeps one MetaOwn member in stable heap-owned Actual, declared BEFORE programs
so reverse destruction keeps it alive through Program teardown. It copies real
Checkpoint metadata, applies the same imported quant override, then finish()s
that retained member unconditionally. Actual is never moved after publication;
returning its unique_ptr moves only the pointer. Numeric/shape/source/catalog
inputs, source admission predicates, kernel selection and CoreV5 are unchanged.

Both --run-actual-metadata and --run-complete-prewalk now check immediately AFTER
returning declare_actual: arch/name/quant pointers equal live owner's c_str(),
all KV pointer arrays/counts and each KV string equal their retained owner,
all4 Program metadata aliases match, then actual GLM profile succeeds. Pointer
ownership is checked BEFORE dereferencing the borrowed arch identity. Checks
repeat before full Reader work and after Reader returns. No dangling string is
read as a diagnostic. Native bind return status is sequenced BEFORE constructing
the message with its completed why, eliminating unspecified argument-order loss.

Unchanged normal targets/schema:
  glm53_main_verified_component_quotes --run-host-quotes
  glm53_main_verified_graph_quotes --run-actual-metadata MODEL EXCLUSIVE_HOME NEW_JSON
  glm53_main_verified_graph_quotes --run-complete-prewalk MODEL EXCLUSIVE_HOME COMPLETE_BUNDLE PROOF ID NEW_JSON
Normal metadata is the fast first gate: it runs all post-return ownership/profile
checks without complete Reader work. The full mode still requires REAL complete
data; no success/capability is forged or substituted from prior receipts.

New explicit read-only mode (same5 arguments as normal metadata):
  glm53_main_verified_graph_quotes --run-native-metadata-diagnostic MODEL EXCLUSIVE_HOME NEW_JSON
It constructs all4 actual metadata Plans and reports exact strategy/dynamic/world,
first violation/count of NativeInventory's original static-unit predicate (tier,
bytes,movable,slab slot,members/optional/storage), tier counts and replay of actual
selected DEVICE layout callbacks with first refusal/status counts and unclaimed
members. It does not call NativeInventory::bind, complete Reader, payload packing,
Session/SourceFactory/Engine or kernel init/launch. The callback equations match
the existing metadata admission predicate; they grant no authority. No Program/
Plan field is normalized or rewritten. Backend allocation/pread counters bracket
the diagnostic itself. Existing real Sampler pinned declaration remains explicit.

Output schema remains glm53-main-verified-components-v1. execution_mode is one of
actual_metadata/complete_prewalk/native_metadata_diagnostic. The latter adds
native_metadata_diagnostic:{schema:glm53-native-authority-metadata-diagnostic-v1,
rank_reports:[4actual logical ranks],Program_or_Plan_normalized:false,
NativeInventory_or_Session_capability_granted:false}. Each row records
native_inventory_bind_called:false,complete_Reader_or_source_scan_called:false.
A diagnostic passed:true means a bounded real-metadata report was published;
nonzero refusal summaries are evidence to inspect, never Native/Engine Ready.
Existing runtime_ready:false and kernel_instances_or_launches:0 remain. Original
Context/CONV refusal bodies and HOST checker are unchanged; counts grow only
from new metadata-owner and diagnostic assertions. Core1535 pins remain0001f0a1.

All build/runtime results are pending parent execution. Author made source/
hash checks only, no compiler/device/model/payload/runtime or production action.

Narrow actual-GLM component source checks (SOURCE-ONLY candidate)

No Core/parent edits, compilation, target execution, GPU use or source tensor
payload read performed by the author. Parent owns every build and invocation.
This package does not implement or admit SourceFactory/Session/Engine, run the
ordinary loader, initialize kernels, or execute a model. It cannot confer Ready.
Frozen matching Core has1535 files, pins SHA256
9986ab23db17a5148caea5e6577620ab68f76bf580aebe7350a4f969baff3ef9.
Configure only against a parent-built archive of that exact source/profile.

Targets / explicit commands:
 glm53_main_verified_component_quotes --run-host-quotes
   Five small HOST pure-quote groups. Actual static Sampler config covers all4
   ranks/real candidate clamp/11rank0 vs8other pinned requests, bounded bad and
   overflowing configs/caller preservation; undeclared member refusal; original
   Tiny Sampler positive arithmetic and profile refusals; unready/invalid loader
   prewalk and Ctx outputs preserved without backend allocation/payload pread.
   These are NOT positive actual NativeSource/loader admission tests.

 glm53_main_verified_graph_quotes --run-actual-metadata \
   PINNED_CHECKPOINT EXCLUSIVE_CACHE_ARCH_HOME NEW_JSON
   Parent fresh-idle AQL metadata gate required. Uses real Checkpoint/cache,
   exclusive actual glm5next_cache declaration, actual Sampler declaration and
   PERSIST output. Sampler pinned allocations are real, prequoted and bounded
   under1MiB total. Instances=false: no kernel initialization/launches, Pools,
   Ctx construction, SourceFactory or model. It quotes real KV metadata, actual
   WTAB/route arrays/maps and tests ten reversible in-memory metadata refusals.
   Four LOGICAL rank graphs selected on one actual gfx1201 eligibility card are
   metadata evidence, not four-card execution/physical-fit evidence.

 glm53_main_verified_graph_quotes --run-complete-prewalk \
   PINNED_CHECKPOINT EXCLUSIVE_CACHE_ARCH_HOME COMPLETE_BUNDLE PROOF EXPECTED_ID NEW_JSON
   Expensive additional parent-only data gate, not an automatic test. Strict
   complete Reader and once/rank actual mapped-source cut verification precede
   opaque rank authorities and actual NativeInventory/final Native plans. No
   fabricated descriptor checksum/SourceReady flag. It then calls ONLY loader
   prepare(), proves all193536 expert views omitted and actual job/maxima/byte
   arithmetic, and cap-minus-one/source-encoding/foreign-plan refusals. Payload
   pread/backend counters are reset around prepare to distinguish zero loader
   prewalk traffic from the separately EXPENSIVE Reader verification traffic.
   No loader execute(), H2D, native payload allocation, Session or Engine.

The unchanged frozen A/B stock_plan.cpp supplies bounded paths/config/JSON helper
bodies; its old main and its cast-hypothesis graph routine are NOT invoked. The
new graph uses actual imports/source cache directly, no encoding override.
Original Tiny Ctx positive tests remain in compact's matching new-Core regression
package glm53-main-verified-source-engine-v5-prototype (unchanged Foundation
contract and actual tiny Engine bodies). This package also pins/rechecks the
exact old Tiny quote bodies in source; it does not duplicate their synthetic graph
and present it as a real-source result. Current strict maxTok1/final-tail/KDA math
is not changed here.

Build (parent only):
 cmake -S HERE -B NEW_BUILD -DRADIANCE_SOURCE=ABS_FROZEN_NEW_CORE \
   -DRADIANCE_BUILD=ABS_MATCHING_WHOLE_CORE -DCMAKE_BUILD_TYPE=Release
 cmake --build NEW_BUILD --target glm53_main_verified_component_quotes \
   glm53_main_verified_graph_quotes --parallel 1
For matching AQL archives add -DRADIANCE_AQL_CORE=ON; HOST basic mode needs none.
Sanitizer flags are copied from matching Core's RAD_ASAN/RAD_TSAN cache; no checks
are disabled. Exact source home/compiler/standard active flags are verified;
unsupported C++ ABI flags refuse. Parent uses its external compiler lock, process
deadline and inherited explicit sanitizer/OMP/environment controls. No services,
ports, production reloads or GPU faults are part of these commands.

V2 preserves the V1 source-only package. Core V2 repairs three mechanically
composed sibling namespace references; all six authored functional Core paths
are unchanged. Component V2 also checks complete-mode path/ID bounds before
metadata/device setup; quote mathematics/test expectations are unchanged.

V3 preserves both componentV1/V2 packages and V2 failed Core compile evidence.
The only CoreV3 change is a PRIVATE composed-source include root in CoreCMake;
all C++ bodies and this package's checkers remain byte-identical toV2. Public
component headers resolve without adding a new include interface.

V4 source-only adaptation: matching Core V4 and direct public Program.device_holes(false).empty() coverage predicate. The old TU-local miss_report was unavailable to this translation unit. All loader, quote, source admission and numerical checks are preserved. Earlier component revisions remain unchanged.

V5 compile-surface repair: only two RAD_MAX_LOGPROBS uses in the HOST component checker are explicitly namespace-qualified as rad::RAD_MAX_LOGPROBS. Core remains exactly V4; arithmetic and guards are unchanged. ComponentV4 and its failed compile evidence remain preserved.

V6 DIAGNOSTIC ONLY: before the FIRST actual Context quote, print one rank0 CTX_QUOTE_DIAGNOSTIC JSON (<=64KiB) containing exact declared arena/scratch/lane fields, every actual KV group geometry/flags/layers (explicit caps128groups/46layers), prepare limits, and the first unsupported schema/deferred row/scratch case from a bounded metadata scan. This reads metadata only: no normalization, Program mutation, retry, new acceptance, owner/Runtime/Engine construction or numerical change. No model pointers/paths/data are logged. The old guards still execute once and may fail; this revision establishes a witness, not a repair. Temporary diagnostic JSON/containers are separate bounded HOST metadata work, not a Ctx storage quote or measured RSS. Removing the helper, one call and its direct include byte-recovers the entire V5 actual checker. V5/failure evidence remain unchanged; Core is still exactV4.

V7 matches frozen CoreV5. It keeps the actual architecture-declared scratch_regions2/lane1scratch0/no-window Program unchanged and checks those positive values before quoting. Scratch refusal now mutates3 rather than the now-genuine2. It locates the one real BF16 CONV6144/nhead1/width4/window0/state0 group without rewriting it, then adds10 one-field tuple refusals: kind,dtype,nhead,head6143/6145,width3/5,window,state0,state1. Each uses existing output/payload/backend-purity checks and restores the POD declaration with RAII. The10 older Ctx refusal cases remain, giving20 metadata refusals in actual-metadata mode. The entire boundedV6 diagnostic is retained. No fabricated positive geometry, runtime/loader/numerical/schedule change or Core edit by this component author. Earlier component/Core revisions and failures remain frozen.
