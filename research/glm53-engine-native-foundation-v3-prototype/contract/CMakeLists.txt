set(TINY_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}" "${RADIANCE_SOURCE}" "${RADIANCE_SOURCE}/core"
  "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/arch/common" "${RADIANCE_SOURCE}/vendor" "${RADIANCE_SOURCE}/tests" "${RADIANCE_HOST_BUILD}/rad_include")
add_library(glm53_tiny_startup_arch MODULE tiny_arch.cpp shape_profile.cpp)
target_compile_features(glm53_tiny_startup_arch PRIVATE cxx_std_20)
target_include_directories(glm53_tiny_startup_arch PRIVATE ${TINY_INCLUDE})
target_compile_options(glm53_tiny_startup_arch PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
set_target_properties(glm53_tiny_startup_arch PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/research-architectures")
# Only the architecture module's normal ABI symbols are bound from the exported
# checker/whole matching Core. No second archive/runtime is linked into the module.
add_executable(tiny_startup_metadata_check checker.cpp fixture_writer.cpp graph_quote.cpp profile.cpp shape_profile.cpp)
target_compile_features(tiny_startup_metadata_check PRIVATE cxx_std_20)
target_include_directories(tiny_startup_metadata_check PRIVATE ${TINY_INCLUDE})
target_compile_options(tiny_startup_metadata_check PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
target_link_libraries(tiny_startup_metadata_check PRIVATE
  "-Wl,--whole-archive" "${RADIANCE_HOST_BUILD}/core/librad_core.a" "-Wl,--no-whole-archive"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_llama.a" "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_httplib.a" Threads::Threads ${CMAKE_DL_LIBS})
target_link_options(tiny_startup_metadata_check PRIVATE "LINKER:--export-dynamic" "LINKER:--wrap=rad_dev_alloc" "LINKER:--wrap=rad_dev_free")
# No CTest, automatic execution, Core target, Engine activation, installation,
# production CLI, GPU/HIP source or service/restart/deployment target.
