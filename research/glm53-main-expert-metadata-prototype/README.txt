GLM5.3 main-expert TP4 metadata adapter — SOURCE ONLY, parent execution pending

This new separate package uses the actual stock Core headers/archive, opened
Checkpoint, resolved CheckpointSources cache, finished Program and actual Plan.
It changes NO Core, frozen parent, serving/config file, kernel or model payload.
The stock v4 planning driver is retained with a narrowly added adapter call once
per logical rank and an actual-input contract matrix on rank0. Parent compiles
and executes; author has run no compiler/checker/device/model/network code.

API:
  int glm53::main_metadata::quote(
    Checkpoint&, const CheckpointSources&, const Program&, const Plan&,
    span<const int32_t> resident_main_units, const Limits&, Quote*, string* why);

The first profile is real stock GLM main MXFP4/E8M0, actual TP4 ranks0..3,
max_spec0, H4096/F2048, E288/top8 and routed layers3..44. The caller's Program and
Plan are borrowed immutable inputs only during the call; no pointer or string
view escapes. No whole WeightInfo snapshot is copied. One bounded name-map index
borrows the actual map strings during the call; no source catalog callback is
accepted. Raw six-tensor gate/up/down code+scale names, order, U8 physical shapes,
byte extents, exact mapped aliases and SAME Checkpoint::import_piece objects are
checked against the actual cache and current maps. A foreign cached span refuses
before importer access. No mapped byte is dereferenced by the adapter; source
payload checksums, immutable file generation or full-payload integrity are NOT
established. Existing upstream Checkpoint/imported metadata lifetimes remain the
caller's responsibility and are outside the adapter-owned metadata quote.

All four members are found in each actual MoveUnit and matched with Program
unit_of_weight/group/slot/encoding/selection/dtype/shape/access/shard/span/rowparts.
Actual weight_share computes ordered rectangles, independently checked as two
ROW halves from distinct gate/up tensors, and per-row K512 DOWN cuts with scale
columns cut by32. Actual Plan offsets, alignments, byte totals and class stride
are re-derived and validated; metadata does not assume stored offsets. Every
actual selected DEVICE main consumer must independently claim canonical layout
via its real layout callback (or lack of a callback). A relayout/tag/widening,
missing DEVICE row, initialized kernel or unsupported geometry refuses; no tag
is erased or canonical shape substituted to make the result pass. The header
formula3342336B is only a conditional cross-check, not a chosen-kernel fit claim.

The result owns geometry-only PODs with actual unit/member IDs, all4 member
sizes/offsets/dtypes/shapes and at most2 actual physical rectangles each. It also
owns index pairs for the nonresident records and an explicit validated main
resident mask. Geometry deliberately lacks NativeRecordDescriptor's generation
and checksum fields; there is no NativeRecordSource, acquire, lease, catalog,
NativeInventory binding, Ready or Engine capability. These are layouts, not
acquirable records. Core-private Native headers from the research fork are NOT
mixed with original stock Plan/Program ABI. Later a matching consumer must bind
actual complete payloads/generation/checksums before any native use.

Actual allocation sizes are quoted with sizeof of these NEW public adapter PODs.
Expected64bit ABI static_asserts are slice80/member176/record880/owned8; parent
compilation establishes them and output reports sizeof values. Exact output
array requests, borrowed-name index, byte-per-weight layout-claim scratch and
bounded stack/temporary scratch are charged BEFORE metadata arrays/source checks.
Two-pass source/layout validation precedes output array construction. The cap is
64MiB by default, never more than128MiB. Output is assigned only after complete
success; all refusals/OOM preserve prior caller storage. This is requested
adapter storage, not upstream graph/Checkpoint memory, malloc overhead, process
RSS, OS page cache, plugin allocator internals, driver memory or a full-model
budget. Caller-owned prior output and caller diagnostic string are separate.

Parent-run actual-input cases (authored, no PASS claim): full geometry/byte/rect
coverage, caller preservation for22 scalar/source/layout/budget mutations,
empty actual-source cache, current swapped GU map, stale swapped GU code and
scale cache with current maps restored, one explicit resident partition and
duplicate resident refusal, and a final positive requiring every mutation
restored. These source/helper API cases neither run a kernel nor write payloads.
Allocator OOM injection is not implemented or claimed; exception preservation
is source-reviewed and the exact metadata-minus-one refusal is explicit.

Driver retains the separately frozen actual stock v4 Planner behavior, including
no sampler declaration and its explicit assumed margins, real R4D metadata
selection,11 nonexpert kv_b BF16 RECIPE HYPOTHESES, no model allocations/queues/
launches or Engine.start. That nonexpert cast does not alter expert source types.
The adapter runs on the first scenario's actual Plan; input_plan_fits is printed
separately, and complete geometry does NOT upgrade a failed placement Plan.
The tool returns nonzero if DEVICE selection, allocation-count invariance, main
metadata, or rank0 contract matrix refuses. Metadata diagnostics are retained in
its NEW report path. No result proves full-context fit, throughput, loaded model
correctness, TP failure coordination or production cache activation.

Parent-only build/run, not executed by author:
  python3 HERE/verify_source_pins.py
  cmake -S HERE -B NEW_BUILD -DRADIANCE_SOURCE=ABS_ORIGINAL_CORE \
    -DRADIANCE_BUILD=ABS_MATCHING_WHOLE_AQL_BUILD -DCMAKE_BUILD_TYPE=Release
  cmake --build NEW_BUILD --parallel 1
  NEW_BUILD/stock_glm53_main_metadata --run-device-metadata \
    --checkpoint ABS_COMPLETE_PINNED_AMD_CHECKPOINT --home ABS_SHIPPED_HOME \
    --output ABS_NEW_JSON --ctx 512 --batch-tokens 128
Matching whole-Core ABI/source pins are checked by CMake.
Only the parent may approve/schedule actual AQL discovery/code eligibility.
There is no automatic execution, CTest, implicit Core build, download/conversion,
installation, serving/deployment or restart target. Metadata source freezes
are distinct from later actual parent compiler/runtime receipts.
