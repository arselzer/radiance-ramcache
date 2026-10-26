# DeepSeek-V4.1-Flash (`deepseek41`): work in progress

This is a working branch, not a finished PR. I'm sharing it so others can test it, read it and help split it. Everything
below was measured on 4× (and earlier 3×) Radeon AI PRO R9700 (gfx1201, 32 GiB each, PCIe Gen5, no P2P bridge),
62 GiB of host RAM.

## What works

- **Architecture plugin** `arch/deepseek41` (DeepSeek-V4.1-Flash-1017, 43 layers incl. the three-stage DSpark drafter):
  - hybrid compressed / sparse attention with the indexer, mHC (`hc_*` mixes), routed + shared experts with the router
    bias, the Engram n-gram tables (looked up on the host, `--ngram-placement disk|ram`), YaRN;
  - any world the head and expert counts allow: TP 1/2/3/4 boot on the host backend, TP3 and TP4 measured on cards.
    The 2304-column shared expert is split by whole 256-column blocks (768/512/512/512 at TP4) where an even split
    would cut a block;
  - DSpark speculation from the release's own `mtp.*` stages (bit-exact against speculation off), and `--spec-ngram`.
- **Two inputs, one plugin:**
  - Unsloth's llama.cpp-layout GGUF (`UD-IQ1_S`), as a conversion source;
  - the official release (`DeepseekV41ForCausalLM`: fp8 e4m3 with an e8m0 scale per 32×32 block, routed experts as
    packed fp4 with an e8m0 per 32 = MXFP4, Engram tables fp8 with an e8m0 per 32 of a row). The reader takes
    `F8_E8M0`, the `X.scale` spelling and the fp4 experts byte for byte; the Engram gate is the product of two release
    tensors (`RAD_MAP_MUL`).
  - Proof for the release path: for every one of the plugin's 32,701 weights, the f32 rad-convert reads from the
    release equals the f32 llama.cpp's converter (`conversion/deepseek41.py`) produces for that tensor: 50.7 G elements,
    0 differences (fp4 code 8 decodes to −0 here and +0 there; no product or sum changes). 95,775 of the release's
    96,085 tensors are read; the other 310 are the vision tower, its aligner, `ffn.gate.bias_vl` and the drafter's
    confidence head.
- **Kernels** (`libr4d`, gfx1201 WMMA): GEMMs for the codebook and K-quant encodings the UD-IQ1_S mix uses (iq1_s, iq2_xxs,
  q5_K, q6_K, q8_0, iq4_nl as affine/grid planes; dense and MoE, prefill and decode heights), sparse attention + indexer,
  a decode-height attention kernel, mHC/router decode kernels, and four fused elementwise chains. All are bit-identical
  to their unfused / reference forms on the gates below.
- **Engine core:** `rad-convert --imatrix` finds llama.cpp's entry names (`.weight` kept; before this no weight matched
  and iq1_s/iq2_xxs were refused); a long host gather's read-ahead runs beside its kernel when its rows are resident.
- **No environment switches.** What the branch adds that a user may set is a flag: `--row-prefetch on|off` (the
  mapped tier's row read-ahead, on), `--mapped-rewarm on|off` (the page-cache rewarm of the mapped tier at startup,
  on), `--debug-tape-dump` (a recorded pass's dispatch list) and `--debug-draft-oracle FILE` (the verify-path
  harness the e2e test drives). The lab's A/B and tuning switches are deleted, each with the value every measured
  run used fixed in code: `deepseek41`'s fused chains and issue order, the `libr4d` kernels' variant and tuning knobs,
  the port's no-quantise and kept-logits diagnostics.

## Numbers (UD-IQ1_S mix, 251 GiB container, greedy, thinking off)

| | TP3 | TP4 |
|---|---|---|
| decode, essay / code edit (tok/s) | ~60 / ~60 | **77.4 / 77.2** |
| decode, code edit with `--spec-ngram 5` (tok/s) | ~95 | ~128 |
| 16k-prompt prefill, Engram rows resident (tok/s) | ~2,500 | **3,746** (ttft 4.24 s) |
| 16k-prompt prefill, Engram rows cold (ttft) | — | ~4.9 s |
| HumanEval@164 / GSM8K@200 | .939 / .975 | .945 / .980 |
| llama.cpp, same GGUF | .939 / .985 | |
| greedy determinism (5 prompts × 32 outputs) | 1 distinct each | 1 distinct each |

TP4 serve: `--tp 4 --tp-wire exact --placement expert_tiered --host-pool-mib 4096 --ngram-placement disk
--max-num-batched-tokens 8192 --checkpoint-interval 8192 --kv-cache-dtype bf16`. The prefill chunk is the smaller of
`--max-num-batched-tokens` and `--checkpoint-interval`, so both are raised together.

## Converting from the official release

`data/recipes/deepseek41-ud-iq1s.recipe` is the UD-IQ1_S per-tensor mix as rules for libquant's `ggml` quantiser
(llama.cpp's own searches, imatrix-weighted), so the release converts to the same encodings the GGUF gives:

```
rad-convert <DeepSeek-V4.1-Flash-1017 release dir> \
  --add engram-tables.safetensors \
  --recipe data/recipes/deepseek41-ud-iq1s.recipe \
  --imatrix <imatrix with the fused names, below> -o deepseek41-ud-iq1s.rad
```

`--plan-only` gives 251.07 GiB. A one-layer slice converts, reads back with `rad-info` and serves on the host backend.
The full conversion and its on-card gates are running now; numbers will follow in the PR thread.

Two inputs are still made outside the tree (see the questions):
- **The imatrix** is Unsloth's (`imatrix_unsloth.gguf_file`, 558 chunks × 4096) plus entries for the plugin's fused
  weights: `ffn_gate_up_{exps,shexp}` = the gate's sums (llama.cpp's gate and up entries are bit-equal: same input) and
  `attn_output_a.g<K>` = columns `[K·4096, (K+1)·4096)` of `attn_output_a` (the grouped matmul's matrices, group-major).
- **The Engram hash tables** (token map, multipliers, bucket sizes, offsets) are computed with llama.cpp's own
  functions from `config.json` and passed with `--add`; they equal the GGUF's.

The release ships no chat template (its format is the Python `encoding/encoding.py`). I serve with this one:

```jinja
{#- DeepSeek-V4.1 chat, plain system/user/assistant text: the official encoding/encoding.py
    format (encode_messages), thinking_mode "chat" (default) or "thinking". No tools, no images. -#}
{%- set mode = "chat" -%}
{%- if thinking_mode is defined and thinking_mode -%}{%- set mode = thinking_mode -%}
{%- elif enable_thinking is defined and enable_thinking -%}{%- set mode = "thinking" -%}{%- endif -%}
{{- "<｜begin▁of▁sentence｜>" -}}
{%- set first_system = messages|length > 0 and messages[0].role == "system" -%}
{%- if mode == "thinking" or first_system -%}{{- "<｜System｜>" -}}{%- endif -%}
{%- if mode == "thinking" -%}{{- "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n" -}}{%- endif -%}
{%- if first_system -%}{{- messages[0].content -}}{%- endif -%}
{%- for m in messages -%}
{%- if loop.index0 == 0 and first_system -%}
{%- elif m.role == "user" -%}{{- "<｜User｜>" + m.content -}}
{%- elif m.role == "system" -%}{{- "<｜System｜>" + m.content -}}
{%- elif m.role == "assistant" -%}{{- "<｜Assistant｜></think>" + m.content + "<｜end▁of▁sentence｜>" -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- "<｜Assistant｜>" + ("<think>" if mode == "thinking" else "</think>") -}}{%- endif -%}
```

Tools and images are not covered.

## Not done

- **`main` does not compile here:** the branch is rebased on `main` at 341cedf, and `main` there includes the
  `core/kvstream/` headers, never committed (#242; the directory is part of #240): neither builds without them.
- **Family gating:** the kernels use RDNA4 WMMA and FP8; each has to be selected on gfx12 only, with the existing op as
  the fallback, and rebased after the RDNA3.5 `libr4d` port lands.
- **This head on cards:** the numbers above are from the commit before the environment switches went. Since then
  `rad-kbench` on one card reads 604 of 604 checked with every per-geometry error unchanged, and 103 of `libr4d`'s 104
  code objects are unchanged; the four IQ1_S / IQ2_XXS decode MoE kernels were recompiled when their debug branch was
  removed. The TP4 output identity and decode speed of this head are not re-run yet.
- **Not gated:** TP2 and TP1 on cards (they boot on the host backend only), thinking mode, contexts past 20k, more than
  4 concurrent sequences, tools, images.

## Planned split

1. `rad-convert --imatrix` name fix (standalone, with its test): open upstream as #267.
2. Long host gather read-ahead beside its kernel (standalone, measured above).
3. DeepSeek-V4 release reader (`F8_E8M0`, 32×32 e8m0 fp8 blocks, MXFP4 experts, `RAD_MAP_MUL`, `rad-convert --sources`).
4. `deepseek41` model support on existing kernels + docs + TP3/TP4 serve configurations.
5. Kernels, one PR per family with measurements on this model and no-regression rows elsewhere.
