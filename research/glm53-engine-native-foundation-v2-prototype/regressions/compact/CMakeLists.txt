cmake_minimum_required(VERSION 3.24)
project(glm53_compact_native_authority LANGUAGES CXX)
set(RESEARCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(RADIANCE_SOURCE "${RESEARCH_ROOT}/radiance-glm-engine-native-foundation-v2-src" CACHE PATH "Matching isolated compact Core")
set(RADIANCE_HOST_BUILD "${RESEARCH_ROOT}/build-glm-engine-native-foundation-v2-host-release" CACHE PATH "Matching compiled HOST Core")
if(NOT EXISTS "${RADIANCE_SOURCE}/core/place/native_weight_authority.cpp")
  message(FATAL_ERROR "Requires the separately authored compact authority source")
endif()
function(compact_standalone TARGET_NAME)
  add_executable(${TARGET_NAME} authority_test.cpp "${RADIANCE_SOURCE}/core/place/native_weight_authority.cpp")
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
  target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${RADIANCE_SOURCE}/core" "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/vendor")
  target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
endfunction()
compact_standalone(compact_authority_test)
compact_standalone(compact_authority_collision_test)
# Test-only source build replaces only the hash accelerator with a constant.
# Full equality/storage/budget algorithms remain unchanged. Real Core never
# defines this macro; no runtime test switch is added to the production API.
target_compile_definitions(compact_authority_collision_test PRIVATE RAD_NATIVE_AUTHORITY_TEST_COLLISIONS=1)

option(COMPACT_STANDALONE_ONLY "Build only pure metadata authority checks, no Core archive" OFF)
if(NOT COMPACT_STANDALONE_ONLY)
  find_package(Threads REQUIRED)
  add_subdirectory(strict_p3)
  # All original unified Native/owner/policy case bodies compile unchanged with
  # the new matching private headers and archive. No frozen source is modified.
  add_subdirectory("${RESEARCH_ROOT}/ds41-unified-native-policy-host-prototype" frozen-unified-regressions)
  add_executable(compact_native_inventory_test inventory_test.cpp)
  target_compile_features(compact_native_inventory_test PRIVATE cxx_std_20)
  target_compile_options(compact_native_inventory_test PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
  target_include_directories(compact_native_inventory_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${RADIANCE_SOURCE}/arch/common" "${RADIANCE_SOURCE}/tools")
  target_link_libraries(compact_native_inventory_test PRIVATE p3_native_source
    "-Wl,--whole-archive" "${RADIANCE_HOST_BUILD}/core/librad_core.a" "-Wl,--no-whole-archive"
    "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_llama.a" "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_httplib.a" Threads::Threads ${CMAKE_DL_LIBS})
  target_link_options(compact_native_inventory_test PRIVATE "LINKER:--wrap=rad_memcpy_async" "LINKER:--wrap=rad_event_record" "LINKER:--export-dynamic")
endif()
# No CTest/auto-run/install/serving/device target. Parent explicitly invokes.
