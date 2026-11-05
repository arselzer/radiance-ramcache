# ds41 native precision and conversion provenance — 2026-11-04

This is a read-only source/receipt audit. No checkpoint payloads were re-read, no numerical tests or builds were run, and no download, GPU, model, or service action was performed. The only write is this report. Frozen P4 source is unchanged.

## Finding

The current `ds41-native-mxfp4-v3` baseline is a **hybrid**: official native MXFP4 routed experts and official backbone tensors, plus ISTA IQ4_NL Engram embedding tables and ISTA integer Engram controls. It must not be described as an entirely untouched official checkpoint or as having no additional quantization anywhere.

The official release itself uses mixed precision, including FP4 routed experts and FP8 dense weights. Retaining those original codes and scales is the relevant native-precision goal; expanding them into BF16 does not restore any precision beyond the published values. The IQ4_NL replacement of the official FP8 Engram tables is a separate, real reduction in precision inherited by our baseline.

Byte-identical replies against our base engine prove preservation of that baseline on the exercised requests. They do not prove equivalence to an independent official runner, nor erase the baseline's Engram substitution. Restoring official Engram requires a new checkpoint baseline and qualification against it, rather than expecting byte identity with the IQ4_NL hybrid.

## Official source and actual import

The source repository is [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170), pinned at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.

The [official configuration at that revision](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json), checked online during this audit, states `quant_method=fp8`, `weight_block_size=[32,32]`, `scale_fmt=ue8m0`, and `expert_dtype=fp4`. Its top-level `dtype=bfloat16` is not a statement that every stored tensor is BF16.

Local provenance roots (unqualified research record names below resolve under `/home/as/radiance-setup/research/deepseek-v41-1017-20261024`, which remains read-only):

- Source metadata: [official-model-info.json](/home/as/radiance-setup/research/deepseek-v41-1017-20261024/ds41-native-mxfp4-source-inventory-20261027/official-model-info.json) and [official-original-index.json](/home/as/radiance-setup/research/deepseek-v41-1017-20261024/ds41-native-mxfp4-source-inventory-20261027/official-original-index.json).
- Authenticated mirror: `/srv/bulk/source-copies/ds41-native-mxfp4/d8b2e3c20f5f4b6d5044d37f20b8d53f260f83e466f72259572be76d7e756bf3-main`. Its `EXPECTED.json` records pinned URLs and expected SHA256 values for originals 01–46; `transport.json` is transport metadata, not a substitute for full SHA authentication.
- The successful conversion is [receipt.json](/home/as/radiance-setup/research/deepseek-v41-1017-20261024/ds41-native-mxfp4-conversion-v3-20261027T121600Z/receipt.json:12). Its exact command is recorded at lines 12–42. It ran on October 27, 12:11–12:22 UTC; the final receipt, including full output SHA readback, finished at 12:35 UTC.

That command used `rad-convert` from `build-ds41-pr266-native-cache-host-required-release-20261027T091000Z`, with `--kernels libref --arch deepseek41`, the native `ds41-native-assets-v3-view-host-20261027T104300Z/input-view`, official tokenizer/config files from `ds41-native-official-tokenizer-metadata-20261027T092900Z`, and this recipe:

```
ds41-pr266-native-hybrid-v2-prototype/native-mxfp4-experts-bf16-nonexperts.recipe
```

The recipe casts named nonexperts to BF16/F32; routed native experts and imported IQ4_NL tables stay unmatched/imported. The command also explicitly set `deepseek41.engram_source=ista_iq4_nl_v1` and supplied these two supplemental files:

```
/srv/bulk/source-copies/ds41/sha256/828c36bfe6f0cdcfd3f7fa8929abcd76f588a3a8da6916d073f0a98a233cc0ee/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf
/srv/models/objects/ds41-engram-iq4nl/eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
```

The output was `native-models/ds41-native-mxfp4-v3-20261027T121600Z/model.rad`, **416,220,274,688 bytes**, SHA256 **`12106cfa7a32c8c6a765c37367639a5cce34dbc26789e484f566d37c4aff25ab`**. Receipt lines 213–240 record output identity, header and full readback. The census records 31,848 logical weights and 62,572 planes, including 30,720 routed expert GU/down weights.

## Supplemental sources: exactly nine logical entries

A bounded scan of the actual conversion's [sources.jsonl](/home/as/radiance-setup/research/deepseek-v41-1017-20261024/ds41-native-mxfp4-conversion-v3-20261027T121600Z/sources.jsonl), selecting pieces with `source_format=supplemental_gguf`, found exactly these nine logical entries:

| Logical entry | Source/encoding | Manifest line |
| --- | --- | ---: |
| `deepseek41.engram.token_map` | IQ2_XS file, I32, stored as is | 1 |
| `per_layer_token_embd.l1.weight` | Engram file, imported IQ4_NL | 4 |
| `per_layer_token_embd.l14.weight` | Engram file, imported IQ4_NL | 5 |
| `blk.1.engram_mult` | IQ2_XS file, I64 slice, stored as is | 820 |
| `blk.1.engram_off` | IQ2_XS file, I64 slice, stored as is | 821 |
| `blk.1.engram_vsz` | IQ2_XS file, I64 slice, stored as is | 822 |
| `blk.14.engram_mult` | IQ2_XS file, I64 slice, stored as is | 11174 |
| `blk.14.engram_off` | IQ2_XS file, I64 slice, stored as is | 11175 |
| `blk.14.engram_vsz` | IQ2_XS file, I64 slice, stored as is | 11176 |

These are six allowed source tensors becoming nine logical entries: a token map, three two-layer integer arrays, and two embedding tables. Current `core/format/ista_engram_source.h:17–22` lists the six source roles. `core/format/checkpoint.cpp:2104–2123` only admits those roles, validates type/shape/ownership and rejects collisions; line 2116 skips unrelated neural tensors. **No floating IQ2_XS expert/backbone weight appears among the supplemental imports.** The IQ2_XS filename is not the precision of our routed experts.

The two table entries retain `affine:codes=u4[1x1],table=i8{1x16},scale=f16[1x32]` (GGML type 20), imported without a further quantization by this conversion. Importing those bytes unchanged does not make them equivalent to the official FP8 tables.

Their pinned origin is [ISTA-DASLab's Engram IQ4_NL file](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/blob/cb02422c900df489cea5833e8a92171224059b3c/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf), revision `cb02422c900df489cea5833e8a92171224059b3c`, SHA256 `eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a`, file size 110,595,290,816 bytes. Local URL/hash provenance is `DS41_ENGINE_QUALIFICATION_LOCAL_ASSETS_20261026.json:40–44`; its initial download flags describe that early inventory's state, not final copy completion. The hot file's `.engram-copy.expected.json` binds its full-SHA source and destination. The conversion receipt holds two authenticated supplements.

The native input-view receipt independently states `original_FP8_Engram_tables_omitted:true` at [receipt.json:14](/home/as/radiance-setup/research/deepseek-v41-1017-20261024/ds41-native-assets-v3-view-host-20261027T104300Z/receipt.json:14).

## What the exactness evidence establishes

For routed experts, actual manifest line 34, for example, records `fp4_e2m1*e8m0[1x32]`, `stored:"as is"`, original w1/w3 FP4 code tensors and their E8M0 scale tensors. The independent conversion census covers all 40×384 GU/down pairs. The pinned import/writer audit records preservation of original FP4E2M1 codes and E8M0 exponent bytes in `DS41_PR266_ORIGINAL_MXFP4_CANONICAL_DISK_QUOTE_SOURCE_ADDENDUM_V1.txt:8–13`. The current writer's corresponding no-rule path at `tools/rad_convert.cpp:1556–1574` writes original source code/scale planes directly. This is an original-plane import, not conversion from IQ2 or another reconstructed expert quantization.

`DS41_NATIVE_MXFP4_CONVERTER_V1_INDEPENDENT_SOURCE_REVIEW_COMPACT.txt:18–25` records the native-as-is census and expressly does not claim a full numerical census or exactness of every BF16 cast. This distinction remains necessary:

- The successful receipt records zero FP32→BF16 source pieces; BF16→F32 pieces are widening conversions. It also records 610 FP8→BF16 source pieces.
- The same receipt explicitly has `exact_BF16_cast_value_claim:false`, `full_value_finiteness_or_exact_BF16_cast_claim:false`, and `official_tokenizer_to_ISTA_controls_semantics_independently_derived:false`.
- Later `ds41_dense_fp8_scan` qualification reports 570/570 eligible dense entries exact, 6,676,480 blocks, zero refused (`tools/ds41-host-experts/README.md:899–915` in the stable tree). That supports exact re-encoding of those current BF16 dense values; it is not a blanket proof for every checkpoint tensor, original raw source plane, or Engram substitution.
- Lossless expert-cache packing and current candidate/oracle checks preserve the values/operation semantics of the imported baseline. They do not independently validate all official model semantics or prove that any other inference engine emits identical tokens.

No new full tensor/scale census was run for this report. Authentication, metadata census, source-level direct-copy evidence, device-oracle coverage and end-to-end reply comparison are distinct evidence types and should remain described separately.

## Restoring official native Engram

Original shards 47 and 48 contain the two official embedding tables. Their small authenticated headers are in `ds41-native-assets-v3-headers-20261027T090200Z/headers/`:

| Layer | Native codes | Native scales | Combined embedding bytes |
| --- | --- | --- | ---: |
| 1 | F8_E4M3 `[384006168,256]` | F8_E8M0 `[384006168,8]` | 101,377,628,352 |
| 14 | F8_E4M3 `[384016682,256]` | F8_E8M0 `[384016682,8]` | 101,380,404,048 |

There is one E8M0 scale per 32 values of each row. The [tail-extraction receipt](/home/as/radiance-setup/research/deepseek-v41-1017-20261024/ds41-native-assets-v3-extract-tail-20261027T095800Z/receipt.json:50) states `original_full_stream_SHA_verified:true` but `full_original_object_stored:false` for both originals (lines 50–52 and 128–130). Only q/k/wkv and wkv scales were retained in the two approximately 157.5 MB `.selected.safetensors` files. Thus the official FP8 embedding payloads are **not available in that local mirror** merely because their complete streams were once authenticated.

Missing original objects at the pinned official revision:

| File | Full bytes | SHA256 |
| --- | ---: | --- |
| `model-00047-of-00048.safetensors` | 101,535,150,936 | `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5` |
| `model-00048-of-00048.safetensors` | 101,537,926,640 | `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd` |

Fetch from `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/1bae0e1088993b9b82de08f59de5be5a34e42170/` plus each filename. A future authorized process can retain the complete objects, or stream-authenticate complete objects while retaining native embedding code/scale tensors with explicit ancestry receipts. This report performs neither action.

Source work needed before a useful native import:

1. Validate/qualify the native FP8/E8M0 table declarations and mappings, retain both original planes, and remove the IQ4 supplemental-table policy from the new conversion inputs. The current recipe already leaves embedding tables unmatched; retain its separate Engram projection/gate cast rules unless an independently justified change is required. Existing official layer joins/names are a starting point, not a qualified complete path.
2. Add an exact native FP8/E8M0 host row-gather path. The existing checkpoint reader at `core/format/checkpoint.cpp:2943–3018` already represents row-group FP8/E8M0 as affine codes plus scales with a `[1,32]` block. Therefore the generic affine-shape test in `arch/deepseek41/deepseek41.cpp:2917–2925` is not by itself evidence of rejection. The clear current runtime blockers are `libref/ref_gemm.cpp:599–602` and `libavx/avx_gemm.cpp:1251–1254`, which accept U4/I4/U8/I8 codes only. The new path needs correct FP8 code and E8M0 exceptional-value/rounding semantics, declarations/registry validation, and host parity gates.
3. Preserve file-backed row access and integrate the code and scale planes with the existing row-prefetch/cache lifetime and accounting rules. Do not materialize either whole table as BF16 or require it to fit RAM/VRAM.
4. Independently derive or verify the integer Engram token-map/hash controls against the official implementation and tokenizer. Copying integer controls unchanged is not independent proof of official semantics.
5. Produce a new authenticated conversion, exercise reference/AVX row gathers and full-model requests, then qualify a new official-native baseline. Benchmark it separately from the old IQ4 hybrid; its changed reply bytes must not be presented as a lossless optimization of the old checkpoint.

Storage sizing from authenticated header offsets and the successful current conversion census:

| Quantity | Bytes | GiB |
| --- | ---: | ---: |
| Current IQ4 Engram payload | 110,595,290,432 | 102.999891 |
| Native FP8/E8M0 Engram payload | 202,758,032,400 | 188.833133 |
| Additional container payload | 92,162,741,968 | 85.833242 |
| Estimated replacement `.rad`, otherwise unchanged | 508,383,016,656 | 473.468580 |

The `.rad` estimate excludes small changes in headers/alignment and is not a fresh conversion measurement. Retaining both complete missing source shards additionally takes 203,073,077,576 bytes. If retained beside both old and new model containers, all copies need storage budget; a future streaming/retention policy must account for its own scratch and receipts.

The larger mapped tables need not increase the configured RAM cache by 85.8 GiB. Preserve variable RAM budgets and file-backed misses. A native embedding row has 264 bytes versus IQ4_NL's 144 bytes, **1.833× payload bytes on a miss**; this is a format-size ratio, not a prediction of end-to-end throughput or physical disk amplification.

## PR266: measured recipe versus runtime capability

ROOT separately verified the live [PR266 API](https://codeberg.org/api/v1/repos/StillDeadcode/radiance/pulls/266): head `acc9ae583099181cf6584f441d745bb377794480`, updated `2026-11-03T12:07:58+01:00`. This section records that ROOT verification, rather than claiming another independent live fetch by this lane.

- [docs/deepseek41-wip.md at that head](https://codeberg.org/StillDeadcode/radiance/src/commit/acc9ae583099181cf6584f441d745bb377794480/docs/deepseek41-wip.md) publishes measurements for a **UD-IQ1_S mixture, 251 GiB**. Those numbers are not measurements of our original-MXFP4 recipe.
- The same work documents an original FP4 input reader and an F32 converter comparison; [docs/deepseek41-hf-direct.md](https://codeberg.org/StillDeadcode/radiance/src/commit/acc9ae583099181cf6584f441d745bb377794480/docs/deepseek41-hf-direct.md) covers the direct-HF path.
- [libr4d/r4d_ds41_mxfp4.inc](https://codeberg.org/StillDeadcode/radiance/src/commit/acc9ae583099181cf6584f441d745bb377794480/libr4d/r4d_ds41_mxfp4.inc) contains native MXFP4 MoE GEMV/tiled rows. The runtime **does support native MXFP4 experts**; a lower-precision published benchmark does not imply absence of that capability or establish whole-model native Engram support.

