Core12 / Fixture16: same-Engine normal4 declaration-context repair

SOURCE ONLY. Frozen Core9/Fixture14 failure and all previous source stay intact.
No compiler, target/checker import/execution, model/GPU/service or Git action by
authors. Matching builds/tests/new actual declaration and model result pending.

Actual Core9 failed RAD_E_SHAPE=-1 at phase3(Plugins) before source admission.
Engine::glm_main_declare passed build_ctx(rank,1), although cfg.max_tok=4.
build_ctx copies its second argument into RadBuildCtx.max_tok and Builder keeps
that context. After the first actual sampler was declared, the existing strict
program.ctx.max_tok!=cfg.max_tok guard refused the one-row Program. The log has
one sampler line, zero geometry/source bytes, and the observed exact status/phase.
input_tokens0 follows because request tokenization is after successful start;
this is not evidence of a tokenizer/template error.

The existing production path changes ONE argument: use cfg_.max_tok. No profile,
Program/route/layout/geometry/sampler/Planner/loader/source/checksum/kernel/math
acceptance guard is relaxed. The defaultR1 still passes1 and normalR4 passes4.
The original metadata/plugin/rank/declaration/geometry prefix is extracted into
private Engine::glm_main_metadata_prepare. Production glm_main_body calls it,
then the unchanged full source factory/remainder. V16_EXTRACTION_SOURCE_PROOF
reconstructs the original C++ byte-exact except the single argument change.
All other Core9 C++ files and all ABI27/architecture/kernel bodies stay exact.

A new internal Harness::declaration_preflight consumes the actual Engine's sole
start attempt and runs that SAME production prefix. It stops BEFORE full Reader,
rank binding/provider, NativeInventory/Plan/Pools, Session, opaque instance init,
streams/Owner/model work. It records actual all4 Program max_tok, each actual
42-map route max_rows, real resolved quantum/attention/state blocks, model Pool
bytes, VMem committed ledger, Registry instances and actual runtime-owner absence.
It retires all owned metadata/Checkpoint/Program/Sampler/plugin resources before
publishing a pointer-free result. Failure leaves output unchanged; either outcome
prevents this same Engine from a later real start. No declaration-only object can
be resumed or treated as source/TransportReady authority.

This is NOT zero driver activity/thread/RSS: actual AQL/HIP/plugin loading and35
Sampler pinned allocations may occur. The actual checker separately observes
rad DEVICE-allocation, stream/event creation and async copy/clear API calls and
requires all zero; pinned call/request counts are reported separately. These
wrappers exclude opaque backend/vendor/driver activity and are not global GPU
instrumentation. No source factory means no Provider payload or full-source
hash; the Checkpoint still maps sources and reads real metadata/tokenizer files.
The no-Ctx/no-appstream/no-instance source path executes no model kernels.
Parent must choose a resource-safe metadata window; no service action is implied.

New explicit targets/modes (never automatic/CTest/install):
HOST: glm53_main_declaration_context_check --run-host-declaration-context
AQL:  glm53_main_declaration_metadata --run-actual-declaration MODEL HOME BUNDLE PROOF ID

HOST has2 authored negative lifecycle groups for requested rows1/4, output bytes
preserved using a VALID aggregate, no source open/read/device alloc/stream calls,
null-output nonattempt, sticky first error and idempotent retirement. It does not
fabricate positive AQL declaration or source authority. The AQL mode uses actual
same-Engine declarations, requires all4 Program/route row capacities4, maps42,
quantum4/attn4/state1, all reported runtime resources absent, retiredtrue and real
start retry refused. Its JSON schema is glm53-actual-same-engine-declaration-context-v1.
The existing manual model CLI/C++ body is byte-exact Fixture14. Existing embedded
prefill_metadata.cpp remains a separate standalone graph query and cannot replace
this same-Engine gate. All previous regression bodies remain exact.

Build WHOLE matching Core12 and plugins in a separate HOME; all3 private fixture
maps must match Core12. No old archive/private-header substitutions. Parent
Release/ASan/TSan HOST checks include the new lifecycle control. Parent should
run/pin the cheap actual same-Engine AQL declaration gate BEFORE another expensive
Reader/prewalk or controlled model retry. No actual model/quality/performance
claim is made by this source repair, and no runtime/source guard is bypassed.
