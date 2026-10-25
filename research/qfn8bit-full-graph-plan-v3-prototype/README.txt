QFN eight-bit experts — HYPOTHETICAL post-conversion metadata quote

NEW standalone source adapted from the parent's passed GLM stock full-graph v4
probe. The author has not compiled, queried devices, loaded plugins or executed
this source. The original Core and every prior prototype are unchanged.

Parent-only build/run against the matching frozen original Core pair:
  cmake -S HERE -B NEW_BUILD -DRADIANCE_SOURCE=ROOT/radiance-src -DRADIANCE_BUILD=ROOT/build-ds41-aql
  cmake --build NEW_BUILD --target qfn8bit_plan --parallel 1
  NEW_BUILD/qfn8bit_plan --run-device-metadata \
    --model /home/as/models/rad/qwen3.8-next-flash-fp8-iq4r-moe.rad \
    --home ROOT/shipped-radiance-1.12.0-home --output ABS_NEW_REPORT.json
Default narrow profile is ctx512/batch128/seq1/MTP0, exact wire, FP8 KV,
F32 linear state, deterministic, embeddingVRAM, no prefix/checkpoints/vision.
Closed optional args: --ctx512|2048|262144, --batch-tokens128|256|2048|4096,
--max-seqs1|4|8, --mtp-depth0|3 (SPACE between name and number, e.g --ctx 2048).
Use a separate fresh output per profile. Production is neither stopped nor changed.
Parent owns fresh health/idle/device/module receipts and every actual invocation.

Metadata hypothesis and authority
- Validates/maps the exact121969901568-byte production .rad for metadata only.
  Full source payload hash/quality is not verified. It never calls file.data or
  read_at and does not construct Checkpoint derived/dequantized payload views.
- Real rad_container_sources supplies every source's rank/shape/encoding. A
  wrapper changes ONLY canonical routed W4NL64 expert logical source encodings
  to E4M3+BF16[128,128]. The exact ten protected BF16 expert identities and all
  non-expert sources stay unchanged. Complete per-rank source coverage is checked.
- The real QFN architecture/layout callbacks choose FP8 ops and TP4 mod4 cuts.
  Actual weight_share checks all selected code/scale rectangles; GU/DN code
  widths must be128/256, GU must retain both stacked halves, and all protected
  views must stay BF16. Sample rectangles/layouts and width histograms are saved.
- This Program does NOT describe the source file's real W4 payload and must
  never be loaded/executed. There is no such code path in the tool. The bool
  PlannerInput.container=true models an eventual matching converted .rad;
  file-tier remains off. Source size is not used as the new native weight budget.

Actual pure planning
Actual Builder instances=false, native metadata row eligibility, arena/scratch
layout, KVManager state/ceiling and BatchBuilder sizes feed the real stock
expert_tiered Planner. No sampler/Engine/Ctx/Pools, streams, uploads, kernel
init_instance or kernel launch are requested. Every resolved instance pointer
must remain null. Device holes and actual reference-only op admission are checked.
Intentional real libavx HOST ngram gather is supported; require shipped libavx.

Each rank's source contains PLE, but physical RAM charges it ONCE. V3 derives
both actual is_ngram_table-classified model members:51200245760B codes and2B
BF16 scalar. Actual Planner raw copy is51200245762B, while weight_footprint
rounds each separate unit:51200249856B. RAM scenarios conservatively add that
page upper once OUTSIDE the four expert pools. All figures and member names
are reported and cross-checked; refusal diagnostics include the actual values.
Disk scenarios add96MiB shared reserve:80MiB row data plus source4.5625MiB set/
reference/clock metadata, rounded conservatively. Physical pages fill on demand;
this is full-capacity arithmetic, not observed usage. TLS readers, OS page cache
and file handles remain in the separately stated illustrative global8GiB margin.
Planner's actual ngRAM copy byte field is checked to avoid accidental4× billing.
Host pools12/14/16/18/20/24/28GiB PER RANK are crossed with assumed runtime/device
margins512/1024/1536/2048MiB and both ngram modes.16MiB/rank sampler host/device
omission margins are explicit. These are conditional budgets, not exact startup
peak/RSS/driver/communicator allocations. Source-only4-card geometry uses one
real visible gfx1201 eligibility card; parent separately verifies identical boards.

No fake numerical rows, no instance initialization, no payload quality result,
no fullmodel inference claim, no fit while the104-GiB production service runs.
HSA discovery/dlopen/static registration can have opaque allocation/driver
effects; unchanged Core tracked allocation bytes do not prove zero process-wide
allocations. Report explicitly records these limits. Output uses O_EXCL,<=16MiB.
completed=true means planning finished; an empty fitting scenario set is useful
evidence, not an inference failure. Device/source admission errors return nonzero.

The second separate task is a tiny converter scale-alias fixture; it is not
invoked here and neither serves nor modifies the production container.

V2 narrow corrections (original v1 preserved): only the real HOST ngram reader
is required, for both disk and locked-RAM copy; DEVICE capability is reported
without requiring a row that stock plugins may not provide. Candidate labeling
requires both MemTotal and the explicit parent-estimated post-stop113.6GiB
host availability, retaining both comparisons and the live snapshot separately.

V3 keeps v1/v2 failed evidence immutable. It derives classified ngram members
and separates raw Planner copy/rounded footprint/host physical page upper.
KV ceiling is worst-case ALL max-seqs simultaneously at FULL configured ctx,
not stock Engine minimum KV admission: a negative c8 result does not imply c8
cannot be configured to share a smaller KV pool. The hypothetical production
trunk/HC/head remain unchanged, unlike a full official-FP8 checkpoint conversion.
