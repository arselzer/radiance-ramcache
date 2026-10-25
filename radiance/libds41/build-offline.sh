#!/usr/bin/env bash
# Compile only. Explicit arch avoids device enumeration. Never load/execute the resulting plugin.
set -euo pipefail
cd "$(dirname "$0")/.."
ds41_build_dir="${1:-build-ds41-offline}"
mkdir -p "$ds41_build_dir"
export ROCR_VISIBLE_DEVICES=99 HIP_VISIBLE_DEVICES=-1
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 -fPIC \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi -shared libds41/ds41_packed_rows.cpp libds41/ds41_latent_host.cpp libds41/ds41_embedding_host.cpp \
  libds41/ds41_compress_finish_host.cpp libds41/ds41_latent_emit_host.cpp libds41/ds41_engram_gate_host.cpp libds41/ds41_qdq_host.cpp \
  libds41/ds41_packed.hip libds41/ds41_latent.hip libds41/ds41_math.hip libds41/ds41_compress.hip \
  libds41/ds41_ffn.hip libds41/ds41_index.hip libds41/ds41_index_paged.hip libds41/ds41_transform.hip libds41/ds41_attention_paged.hip libds41/ds41_route_jobs.hip libds41/ds41_index_stream.hip libds41/ds41_embedding.hip libds41/ds41_reorder_heads.hip libds41/ds41_status.hip \
  libds41/ds41_compress_finish.hip libds41/ds41_latent_emit.hip libds41/ds41_engram_gate.hip libds41/ds41_qdq.hip \
  -o "$ds41_build_dir/libds41.so"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi libds41/ds41_packed_selftest.cpp libds41/ds41_packed_host.cpp libds41/ds41_latent_host.cpp -ldl \
  -o "$ds41_build_dir/ds41-packed-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi libds41/ds41_math_selftest.cpp -ldl -o "$ds41_build_dir/ds41-math-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi libds41/ds41_compress_selftest.cpp -ldl -o "$ds41_build_dir/ds41-compress-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi libds41/ds41_ffn_selftest.cpp -ldl -o "$ds41_build_dir/ds41-ffn-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi libds41/ds41_index_selftest.cpp -ldl -o "$ds41_build_dir/ds41-index-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi -I vendor libds41/ds41_transform_selftest.cpp -ldl -o "$ds41_build_dir/ds41-transform-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi -I vendor libds41/ds41_attention_paged_selftest.cpp -ldl -o "$ds41_build_dir/ds41-attention-paged-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi libds41/ds41_route_jobs_selftest.cpp libds41/ds41_route_jobs.hip -o "$ds41_build_dir/ds41-route-jobs-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi libds41/ds41_index_stream_selftest.cpp libds41/ds41_index_stream.hip -o "$ds41_build_dir/ds41-index-stream-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi libds41/ds41_status_selftest.cpp -ldl -o "$ds41_build_dir/ds41-status-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi libds41/ds41_engram_hip.cpp libds41/ds41_engram_hip_selftest.cpp -pthread \
  -o "$ds41_build_dir/ds41-engram-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi -I vendor libds41/ds41_embedding_selftest.cpp libds41/ds41_embedding.hip libds41/ds41_embedding_host.cpp \
  -o "$ds41_build_dir/ds41-embedding-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi libds41/ds41_reorder_heads_selftest.cpp libds41/ds41_reorder_heads.hip libds41/ds41_reorder_heads_host.cpp \
  -o "$ds41_build_dir/ds41-reorder-heads-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi -I vendor libds41/ds41_compress_finish_selftest.cpp libds41/ds41_compress_finish.hip libds41/ds41_compress_finish_host.cpp \
  -o "$ds41_build_dir/ds41-compress-finish-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi -I vendor libds41/ds41_latent_emit_selftest.cpp libds41/ds41_latent_emit.hip libds41/ds41_latent_emit_host.cpp \
  -o "$ds41_build_dir/ds41-latent-emit-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi -I vendor libds41/ds41_engram_gate_selftest.cpp libds41/ds41_engram_gate.hip libds41/ds41_engram_gate_host.cpp \
  -o "$ds41_build_dir/ds41-engram-gate-selftest"
nice -n 19 hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 \
  -fno-fast-math -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
  -I abi libds41/ds41_qdq_selftest.cpp libds41/ds41_qdq.hip \
  -o "$ds41_build_dir/ds41-qdq-selftest"
