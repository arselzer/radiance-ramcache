cmake_minimum_required(VERSION 3.24)
project(radiance_ramcache_portable_host LANGUAGES CXX)
find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
get_filename_component(PORTABLE_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(RADIANCE_SOURCE "${PORTABLE_REPO_ROOT}/radiance" CACHE PATH "Exported selected GLM main TP4 integration source")
set(RADIANCE_HOST_BUILD "" CACHE PATH "Explicit separately built matching HOST Core")
set(PORTABLE_MODE "release" CACHE STRING "release | asan | tsan")
if(NOT PORTABLE_MODE MATCHES "^(release|asan|tsan)$")
  message(FATAL_ERROR "Unknown portable HOST profile")
endif()
if(NOT IS_ABSOLUTE "${RADIANCE_HOST_BUILD}")
  message(FATAL_ERROR "Pass an absolute matching RADIANCE_HOST_BUILD")
endif()
file(REAL_PATH "${RADIANCE_SOURCE}" PORTABLE_CORE_REAL)
file(REAL_PATH "${PORTABLE_REPO_ROOT}/radiance" PORTABLE_EXPECTED_REAL)
if(NOT PORTABLE_CORE_REAL STREQUAL PORTABLE_EXPECTED_REAL)
  message(FATAL_ERROR "Private fixture headers must come from the exported radiance/ tree")
endif()
# NEW repository-normalized verification, not the unchanged archival CMake pin
# loops which intentionally describe older whole source trees and two pyc files.
execute_process(COMMAND "${Python3_EXECUTABLE}" "${PORTABLE_REPO_ROOT}/tools/verify_snapshot.py"
  --repo-root "${PORTABLE_REPO_ROOT}" --existing-core-build "${RADIANCE_HOST_BUILD}" --mode "${PORTABLE_MODE}"
  RESULT_VARIABLE VERIFY_STATUS)
if(NOT VERIFY_STATUS EQUAL 0)
  message(FATAL_ERROR "Portable source/private-Core verification failed")
endif()
if(PORTABLE_MODE STREQUAL "asan")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
elseif(PORTABLE_MODE STREQUAL "tsan")
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()
set(FOUNDATION "${PORTABLE_REPO_ROOT}/research/glm53-engine-native-foundation-v3-prototype")
set(TRANSPORT "${PORTABLE_REPO_ROOT}/research/glm53-tiny-engine-native-transport-v4-prototype")
set(TP4 "${PORTABLE_REPO_ROOT}/research/glm53-main-tp4-cache-integration-v4-prototype")
set(COMPACT "${FOUNDATION}/regressions/compact")
set(P3 "${COMPACT}/strict_p3")
set(PROVIDER "${PORTABLE_REPO_ROOT}/research/glm53-sharded-native-provider-prototype")
set(UNIFIED "${PORTABLE_REPO_ROOT}/research/ds41-unified-native-policy-host-prototype")
set(S1 "${PORTABLE_REPO_ROOT}/research/glm53-expert-record-store-prototype")
set(BRIDGE "${PORTABLE_REPO_ROOT}/research/glm53-record-cache-bridge-prototype")
set(CACHE_SOURCE "${PORTABLE_REPO_ROOT}/research/glm53-cache-prototype")
set(CORE_INCLUDES "${RADIANCE_SOURCE}/core" "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/vendor" "${RADIANCE_HOST_BUILD}/rad_include")
set(CORE_LINK "-Wl,--whole-archive" "${RADIANCE_HOST_BUILD}/core/librad_core.a" "-Wl,--no-whole-archive"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_llama.a" "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_httplib.a"
  Threads::Threads ${CMAKE_DL_LIBS})
function(portable_compile TARGET_NAME)
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
  target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
  target_include_directories(${TARGET_NAME} PRIVATE ${CORE_INCLUDES})
endfunction()
function(portable_core_exe TARGET_NAME)
  portable_compile(${TARGET_NAME})
  target_link_libraries(${TARGET_NAME} PRIVATE ${CORE_LINK})
  target_link_options(${TARGET_NAME} PRIVATE "LINKER:--export-dynamic")
endfunction()

add_executable(tiny_native_foundation_check "${FOUNDATION}/checker.cpp")
portable_core_exe(tiny_native_foundation_check)
target_link_options(tiny_native_foundation_check PRIVATE "LINKER:--wrap=rad_dev_alloc" "LINKER:--wrap=rad_dev_free"
  "LINKER:--wrap=rad_memcpy_async" "LINKER:--wrap=rad_event_record" "LINKER:--wrap=rad_event_sync")

set(CONTRACT "${FOUNDATION}/contract")
set(CONTRACT_INCLUDES "${CONTRACT}" "${RADIANCE_SOURCE}" "${RADIANCE_SOURCE}/arch/common" "${RADIANCE_SOURCE}/tests")
add_library(glm53_tiny_startup_arch MODULE "${CONTRACT}/tiny_arch.cpp" "${CONTRACT}/shape_profile.cpp")
portable_compile(glm53_tiny_startup_arch)
target_include_directories(glm53_tiny_startup_arch PRIVATE ${CONTRACT_INCLUDES})
set_target_properties(glm53_tiny_startup_arch PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/research-architectures")
add_executable(tiny_startup_metadata_check "${CONTRACT}/checker.cpp" "${CONTRACT}/fixture_writer.cpp"
  "${CONTRACT}/graph_quote.cpp" "${CONTRACT}/profile.cpp" "${CONTRACT}/shape_profile.cpp")
portable_core_exe(tiny_startup_metadata_check)
target_include_directories(tiny_startup_metadata_check PRIVATE ${CONTRACT_INCLUDES})
target_link_options(tiny_startup_metadata_check PRIVATE "LINKER:--wrap=rad_dev_alloc" "LINKER:--wrap=rad_dev_free")
add_dependencies(tiny_startup_metadata_check glm53_tiny_startup_arch)

foreach(TARGET_NAME IN ITEMS compact_authority_test compact_authority_collision_test)
  add_executable(${TARGET_NAME} "${COMPACT}/authority_test.cpp" "${RADIANCE_SOURCE}/core/place/native_weight_authority.cpp")
  portable_compile(${TARGET_NAME})
  target_include_directories(${TARGET_NAME} PRIVATE "${COMPACT}")
endforeach()
target_compile_definitions(compact_authority_collision_test PRIVATE RAD_NATIVE_AUTHORITY_TEST_COLLISIONS=1)

add_library(portable_p3_native_source STATIC "${P3}/native_source.cpp" "${P3}/rank_native_store.cpp"
  "${PROVIDER}/candidate_native_record_store.cpp")
portable_compile(portable_p3_native_source)
target_include_directories(portable_p3_native_source PUBLIC "${P3}" "${PROVIDER}" ${CORE_INCLUDES})
add_executable(glm53_native_source_strict_policy_test "${P3}/native_source_test.cpp")
portable_core_exe(glm53_native_source_strict_policy_test)
target_include_directories(glm53_native_source_strict_policy_test PRIVATE "${RADIANCE_SOURCE}/arch/common" "${RADIANCE_SOURCE}/tools")
target_link_libraries(glm53_native_source_strict_policy_test PRIVATE portable_p3_native_source)
target_link_options(glm53_native_source_strict_policy_test PRIVATE "LINKER:--wrap=rad_memcpy_async" "LINKER:--wrap=rad_event_record"
  "LINKER:--wrap=rad_event_wait" "LINKER:--wrap=pread")

add_executable(compact_native_inventory_test "${COMPACT}/inventory_test.cpp")
portable_core_exe(compact_native_inventory_test)
target_include_directories(compact_native_inventory_test PRIVATE "${COMPACT}" "${RADIANCE_SOURCE}/arch/common" "${RADIANCE_SOURCE}/tools")
target_link_libraries(compact_native_inventory_test PRIVATE portable_p3_native_source)
target_link_options(compact_native_inventory_test PRIVATE "LINKER:--wrap=rad_memcpy_async" "LINKER:--wrap=rad_event_record")

add_library(portable_legacy_native_source STATIC "${S1}/native_record_store.cpp" "${BRIDGE}/record_cache_bridge.cpp" "${CACHE_SOURCE}/expert_cache.cpp")
portable_compile(portable_legacy_native_source)
target_include_directories(portable_legacy_native_source PUBLIC "${S1}" "${BRIDGE}" "${CACHE_SOURCE}" ${CORE_INCLUDES})
add_library(unified_policy_collector MODULE "${UNIFIED}/collector_fixture.cpp")
portable_compile(unified_policy_collector)
add_executable(ds41_unified_native_policy_host_test "${UNIFIED}/checker.cpp")
portable_core_exe(ds41_unified_native_policy_host_test)
target_include_directories(ds41_unified_native_policy_host_test PRIVATE "${RADIANCE_SOURCE}/arch/common")
target_compile_definitions(ds41_unified_native_policy_host_test PRIVATE UNIFIED_POLICY_COLLECTOR_MODULE="$<TARGET_FILE:unified_policy_collector>")
target_link_libraries(ds41_unified_native_policy_host_test PRIVATE portable_legacy_native_source)
target_link_options(ds41_unified_native_policy_host_test PRIVATE "LINKER:--wrap=rad_memcpy_async" "LINKER:--wrap=rad_event_record")
add_dependencies(ds41_unified_native_policy_host_test unified_policy_collector)
# Eighth checker: unchanged v4 body and original caller-owned fixture writer.
# Engine's fixed source-factory helpers already live in isolated namespaces in
# the matching Core; do not link a second source-factory/provider archive here.
add_executable(tiny_engine_native_transport_check "${TRANSPORT}/checker.cpp"
  "${TRANSPORT}/contract/fixture_writer.cpp" "${TRANSPORT}/contract/profile.cpp"
  "${TRANSPORT}/contract/shape_profile.cpp")
portable_core_exe(tiny_engine_native_transport_check)
target_include_directories(tiny_engine_native_transport_check PRIVATE
  "${TRANSPORT}" "${TRANSPORT}/contract" "${RADIANCE_SOURCE}")
target_link_options(tiny_engine_native_transport_check PRIVATE
  "LINKER:--wrap=rad_dev_alloc" "LINKER:--wrap=rad_dev_free"
  "LINKER:--wrap=rad_memcpy_async" "LINKER:--wrap=open" "LINKER:--wrap=pread")
# The shared architecture module's tiny_arch/shape_profile bodies are byte-exact
# to v4's contract copies (recorded in staging provenance), so reuse that module.
add_dependencies(tiny_engine_native_transport_check glm53_tiny_startup_arch)
# No archival add_subdirectory/pin rewrite, CTest/automatic execution, GPU/HIP,
# optional physical exporter target, install, production CLI activation or serving.
# The eighth explicit manual checker calls actual tiny HOST Engine.start only.

# Ninth and tenth manual checks use the source-identical reviewed TP4 fixture V4.
# They do not activate a real TP4 source or execute a model/GPU step.
add_executable(glm53_tp4_cache_session_check "${TP4}/session_checker.cpp")
portable_core_exe(glm53_tp4_cache_session_check)
target_include_directories(glm53_tp4_cache_session_check PRIVATE "${TP4}")
add_executable(glm53_tp4_mover_fence_check "${TP4}/mover_wait_checker.cpp")
portable_core_exe(glm53_tp4_mover_fence_check)
target_include_directories(glm53_tp4_mover_fence_check PRIVATE
  "${TP4}" "${RADIANCE_SOURCE}/arch/common" "${RADIANCE_SOURCE}/tools")
target_link_libraries(glm53_tp4_mover_fence_check PRIVATE portable_p3_native_source)
target_link_options(glm53_tp4_mover_fence_check PRIVATE
  "LINKER:--wrap=rad_memcpy_async" "LINKER:--wrap=rad_event_record" "LINKER:--wrap=rad_stream_sync")
