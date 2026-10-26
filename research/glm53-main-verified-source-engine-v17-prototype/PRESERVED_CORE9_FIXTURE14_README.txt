CoreV7 / FixtureV9: real padded GLM vocabulary guard and sampler mask

SOURCE ONLY. No compiler, checker/model/Torch/GPU/service execution by author.
All frozen CoreV6/FixtureV8/ComponentsV12/RunnerV4 and their evidence stay intact.
The first actual attempt FAILED before source admission: tokenizer154856 versus
model154880 tripped the new manual equality guard (RAD_E_FORMAT=-14), phase1.
No GLM token was generated. Parent restored QFN and independently checked it.
This repair has not been run. No second GPU window is scheduled/authorized here.

Small scope
Model, embedding and head stay154880 rows; TP4 local width stays38720. The SAME
production admission now verifies actual owned metadata/tokenizer/view counts,
raw BF16 embedding/head physical headers154880x4096 with exact byte spans, and
special/EOG IDs below154856. It admits exactly the released prefix154856 and24
padded tail rows, not an arbitrary smaller vocabulary. Output is a pointer-free
POD assigned only after success; a pure facts guard is testable without fake
payload pointers. Actual runtime facts come from the Engine-owned Checkpoint,
Vocab and TokenView. No mask substitutes for source/full-bundle identity checks.

SamplerConfig.valid_token_count defaults0: all stock/default/tiny paths retain
their old behavior. The GLM manual Engine alone opts in after successful actual
admission. The opt-in declaration is closed to154880/154856, world4/rank0..3,
local38720/proper offset, seq1/spec0/no draft coupling. It uses already declared
sample_mask/staging, with no new op/schema/kernel or model-arithmetic change.
At build_step end it ANDs existing grammar/reasoning masks with the real-ID prefix
(or starts from that prefix when unconstrained). build_greedy refreshes every
current row on EVERY call, including greedy_ready after a grammar build; no stale
mask resurrection. TP4 greedy uses the existing masked top1 chain. Rank0..2 allow
all38720 local columns. Rank3 allows38696 and clears24; its final word is0xFF.
Prompt and final output checks use valid154856 instead of padded154880.

Memory/private ABI
No extra pinned/device mask plane or upload staging allocation is added. The
internal SamplerConfig and Engine snapshots gain owned scalar fields; real sizeof
quotes compile against matching CoreV7, including embedded Sampler size. Existing
logical metadata/control allowances remain; full matching Core is required.
ABI27 public headers/schema/plugin exports are unchanged. Old Core archives with
new private headers are refused by exact inventory/home guards. All three active
fixture maps (top/device/strictP3) must equal new whole CoreV7 pins. Defaults and
legacy tests are preserved; no current production/default serving activation.

New target: glm53_main_vocabulary_check (HOST Core ONLY)
No CTest/default execution/install action. Explicit parent modes:
  BUILD/glm53_main_vocabulary_check --run-host-vocab MATCHING_CORE7_HOME
  BUILD/glm53_main_vocabulary_check --model-vocab-check ACTUAL_CHECKPOINT_DIRECTORY

The first has three authored groups/31 refusal cases: pure metadata/range/output
preservation, actual Sampler staging on all4 logical ranks, grammar intersection
and cached-greedy refresh, opt-in refusals and default0 control. It launches the
unchanged HOST sample_mask and sample_topk rows directly through actual loaded
ABI metadata: the24 padded logits deliberately exceed every valid logit. They
cannot win under the mask; the old unmasked control DOES select a padded ID.
The grammar VocabView is an explicitly synthetic test table; no model/source or
TP collective is executed or simulated as a pass. Existing inherited suites
remain separate. Pinning says authored, never invents measured counts.

The second invokes the actual Engine read_metadata/load_vocab routines plus the
SAME production admission, without start(), plugin loading, source admission,
Plan/Pools/Ctx/Sampler/queue or model construction. It loads real headers and the
real tokenizer/config/generation sidecars; it maps checkpoint virtual ranges as
the stock Checkpoint does, but does not scan weight payload. Count wrappers require
zero calls to device count/set/props/alloc/stream creation over the entire call,
including destruction. It returns a fully owned POD after source owners close.
This CPU check can and should run while QFN serves, before another maintenance
window. It does not authenticate payload bytes or replace the trusted72-file
source proof/tokenizer-artifact binding, full-source prewalk or GPU correctness.
Use exact HOST Core, hidden GPUs, parent heavy lock and available CPU memory;
stock parser/GBNF/VA/allocator costs are not converted into an RSS quota here.

Parent build/admission handoff
- New source radiance-glm-main-verified-source-v7-src, new FixtureV9 ONLY.
- Add strict parent builder/validator pair Core v7 -> fixture v9, and separate
  fixture-v9 build directory; retain previous v6->v8 as historical matching pair.
- Build the WHOLE matching HOST Core and plugins first. Exact Core7/header/archive
  HOME must match, including changed SamplerConfig layout. No old private ABI reuse.
- Existing target names remain; additionally build glm53_main_vocabulary_check.
  Record/hash its executable alongside manual/gate/regression binaries in the
  parent build/validation receipts. Run --run-host-vocab before the legacy suite.
- Then run --model-vocab-check on the real checkpoint with exact CPU build; require
  status0/model154880/tokenizer154856/padded24/local38720/backend_calls0. Bind log,
  binary/source hashes and actual tokenizer/header artifacts in a NEW receipt.
- Independent source review + matching Release/ASan/TSan and real CPU preflight
  precede considering AQL metadata/full prewalk or any separately controlled run.
  This source handoff requests no service stop/GPU retry and adds no runner bypass.

Historical parent README is preserved separately and does not override these
source revisions, targets or pending status. The new mask path requires actual
DEVICE row resolution in any future AQL gate; HOST proof is not GPU proof.
