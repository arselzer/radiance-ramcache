# Opt-in experimental kernels/oracles. Never install or place them in the default serving scan.
set(RAD_PLUGIN_INSTALL OFF)
rad_add_plugin(libds41_ref experimental/kernels
  SOURCES ds41_packed_rows.cpp ds41_packed_host.cpp ds41_latent_host.cpp ds41_host_ops.cpp ds41_embedding_host.cpp ds41_reorder_heads_host.cpp
  ds41_compress_finish_host.cpp ds41_latent_emit_host.cpp ds41_engram_gate_host.cpp ds41_qdq_host.cpp)
target_compile_definitions(libds41_ref PRIVATE DS41_HOST_REFERENCE=1)
target_compile_options(libds41_ref PRIVATE -fno-fast-math -ffp-contract=off)
if(RAD_HIP_FOUND)
  rad_plugin_gpu_targets(DS41_GPU_TARGETS "^gfx1201$")
  if(DS41_GPU_TARGETS)
    rad_add_plugin(libds41 experimental/kernels
      SOURCES ds41_packed_rows.cpp ds41_latent_host.cpp ds41_embedding_host.cpp
      ds41_compress_finish_host.cpp ds41_latent_emit_host.cpp ds41_engram_gate_host.cpp ds41_qdq_host.cpp
      HIP_SOURCES ds41_packed.hip ds41_latent.hip ds41_math.hip ds41_compress.hip
      ds41_ffn.hip ds41_index.hip ds41_index_paged.hip ds41_transform.hip ds41_attention_paged.hip ds41_route_jobs.hip ds41_index_stream.hip ds41_embedding.hip ds41_reorder_heads.hip ds41_status.hip
      ds41_compress_finish.hip ds41_latent_emit.hip ds41_engram_gate.hip ds41_qdq.hip GPU_TARGETS ${DS41_GPU_TARGETS})
    target_compile_options(libds41 PRIVATE -fno-fast-math -ffp-contract=off $<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64>)
    set_target_properties(libds41 PROPERTIES HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
  endif()
endif()
