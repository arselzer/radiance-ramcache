# DS41 drafter quantization recipe, 8 November 2026

The [README table](../README.md#drafter-quantization-pilot-8-november) compares the original
MXFP4 drafter with two smaller versions. This document records how those artifacts were made.
The conversion tools currently live in the frozen research packets listed below; they are not
a merged `radiance convert` feature. Commands refer to the research server and existing builds.

## Source and scope

Start from the DSpark bank extracted from this project's native-MXFP4 1017 import:

```text
/home/as/radiance-setup/research/deepseek-v41-1017-20261024/native-models/ds41-dspark-bank-20261028T140400Z/draft.rad
SHA256 a497bbf97997a179ce3e08fb8908f16ba70e14ad8a4c432f361cbcbef20f1bdd
```

It contains three drafter MoE stages, layers 40–42, with 128 experts each. All 384 expert IDs,
routers and dense weights remain. Every expert has a fused gate/up matrix of 4,608 × 5,120 and
a down matrix of 5,120 × 2,304. The target model is never opened or modified by the converter.
Its checkpoint precision is described in the root README; these experiments change only draft
proposals, which the unchanged target verifies.

Two sequential conversions produced the tested banks:

1. Decode the native MXFP4 gate/up rows into F32 batches and encode all 384 gate/up matrices as
   **IQ2_XXS**, retaining the native MXFP4 down matrices and every dense plane.
2. Starting from that artifact, decode and encode all 384 down matrices as **IQ2_S**, copying
   the existing IQ2_XXS gate/up bytes and all dense planes unchanged. GU is not requantized.

The IQ2_XXS blocks store 256 values in 66 bytes; IQ2_S stores them in 82 bytes. These are the
specific formats tested, not IQ2_XS or an unspecified two-bit format. All importance weights
were explicitly **1.0**: `UNCALIBRATED_UNIFORM_IMPORTANCE`. No activation calibration, measured
importance matrix, fine-tuning, expert pruning or target quantization was used. The all-ones
weighted path is distinct from a null-importance invocation.

## Encoder and execution

The HIP encoder comes from [trench-gguf, pinned revision
74243909daf8eb40c9c603dba2133e80031d2b8d](https://github.com/zephel01/trench-gguf/tree/74243909daf8eb40c9c603dba2133e80031d2b8d),
with its MIT notice retained. The selected IQ2 routines are literal upstream bodies. CPU
quantization/dequantization and table initialization use the same-candidate Radiance GGML port
`lq_ggml_port.cpp` (CPU authority `06938ac12`). IQ2_S uses GGML type 22 and its own 1,024-entry
grid and initialized map/neighbour tables; IQ2_XXS tables are not reused for it.

Only one RX 9070 XT performed offline quantization. Rows were processed in batches of 1,024,
with bounded F32 scratch rather than a full F32 bank. Native source decoding, output writing,
hashing, error diagnostics and readback stayed on the CPU. The complete DN conversion, including
these costs, took 414.41 seconds. That is not GPU kernel time or a forecast for another bank.

Frozen source locations under `/home/as/ds41-claude-build/work/`:

| Component | Source packet / file |
|---|---|
| Gate/up GPU converter | `draft-iq2-gpu-converter-source-v4-20261107/gpu_converter.cpp` |
| IQ2_XXS encoder | `draft-iq2-trench-iq2-only-source-20261107/` |
| Final gate/up bank packaging | `gu-experiment-artifact-root2-20261107/producer-command.json` records the executed command |
| Down GPU converter | `draft-iq2s-dn-gpu-converter-source-20261108/gpu_converter.cpp` |
| IQ2_S encoder, CPU bridge and fixtures | `draft-iq2s-encoder-source-20261108/` |
| DN converter build recipe | `draft-iq2s-dn-converter-build-source-v2-20261108/` |
| Owned DN conversion window | `draft-iq2s-dn-conversion-window-source-20261108/` |

The GPU converter interfaces used were:

```bash
# Gate/up only: native bank -> IQ2_XXS GU, original MXFP4 DN.
"$GPU_CONVERTER" --gpu-full \
  "$NATIVE_BANK" "$NATIVE_SHA256" "$NEW_BANK" "$REPORT" gu 1024 \
  "$XXS_HIP_LIBRARY" "$XXS_CPU_TABLE" \
  "$LIVE_RECEIPT" "$LIVE_RECEIPT_SHA256" --uniform-uncalibrated

# Down only: existing IQ2_XXS GU/native DN -> IQ2_XXS GU/IQ2_S DN.
"$DN_CONVERTER" --gpu-dn-iq2-s \
  "$GU_BANK" "$GU_SHA256" "$NEW_BANK" "$REPORT" dn 1024 \
  "$S_HIP_LIBRARY" "$S_CPU_TABLE" \
  "$LIVE_RECEIPT" "$LIVE_RECEIPT_SHA256" --uniform-uncalibrated
```

These are the converter argument layouts, not permission to run outside an owned window.
The executables require a freshly issued receipt tied to the live owner/session, both research
locks, stopped QFN, an empty five-GPU client inventory, the selected device and finite limits.
Old receipts cannot be replayed. The owner sets the matching `RADIANCE_DS41_*` receipt variables
and inherited FD9. On this server, launch a freshly prepared DN configuration through:

```bash
W=/home/as/ds41-claude-build/work
bash "$W/draft-iq2s-dn-conversion-window-source-20261108/native-owner.sh" \
  "$FRESH_CONFIG" "$FRESH_CONFIG_SHA256" "$NEW_RUN_DIRECTORY"
```

The wrapper stops QFN, executes the converter, joins its complete session, verifies device
retirement, restores the same QFN container/image and checks its normal reply. Source/build
pins and a new output directory must be prepared for each run. Builds follow
`work/AGENT-LOAD-RULES.md`; these are not standalone commands for an arbitrary checkout.

For the completed DN run, the executable and encoder inputs were:

```text
work/gpu-resume-kv-j1-20261106/IQ2S-DN-converter-build-root1/component
work/gpu-resume-kv-j1-20261106/IQ2S-encoder-build-root2/libgpuq.so
work/gpu-resume-kv-j1-20261106/IQ2S-encoder-build-root2/iq2-tables.bin
```

The final GU/native-DN bank was separately packaged from the GU conversion while preserving
the original native lineage and ordinary metadata. Its executed command and full source/output
mapping are retained in `gu-experiment-artifact-root2-20261107/`; this packaging is part of the
recipe, rather than renaming an intermediate prototype to a serving bank.

## Outputs and checks

| Artifact | File bytes | SHA256 |
|---|---:|---|
| IQ2_XXS GU / MXFP4 DN | 6,205,222,912 | `96b0343143511af9e5e578e542aa3f22e52d57d1596d3464f4e54805af515d27` |
| IQ2_XXS GU / IQ2_S DN | 5,289,811,972 | `f656ec3acc59ff17124b1f844e4cdd77f77b2804369b4c55dcf60943292b0e68` |

RAD file sizes differ from loaded expert payload because the container carries its own plane,
scale and metadata representation. The latter bank contains 902.8125 MiB of routed expert
payload per rank; its aligned GPU reservation is 903.0 MiB/card. Dense weights, KV and transient
buffers are additional.

Before full DN conversion, 37 IQ2_S CPU/GPU encoder comparisons passed: 32 synthetic, three real
down-weight samples and two batch-throughput cases. Every encoded byte had to equal the CPU
reference. Full conversion checked all 384 down matrices and all 4,529,848,320 source values,
native pack/unpack bytes, and CPU/Core dequantized words. All 384 GU matrices and every dense
plane were copied and checked unchanged. Source and output full hashes and file generations,
all 768 expert directory entries, names, IDs, shapes and ordinary metadata were checked.

The full bank was **not** independently re-encoded by the CPU: sample encoder parity and full
bank dequantization/structural checks have different scopes. Separate four-card native controlled
MoE tests and all 80 CODE pilot reply/count checks passed. The remaining broad/state/hash model
gates and reverse-order timing checks are still required before adoption. No error metric is
substituted for exact target replies. Full local evidence and historical controls are in
`/home/as/ds41-decode-review-20261028/DRAFTER-PLACEMENT-RESULTS-20261108.md`.
