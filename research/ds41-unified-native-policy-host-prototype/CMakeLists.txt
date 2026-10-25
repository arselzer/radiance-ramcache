cmake_minimum_required(VERSION 3.24)
project(ds41_unified_native_policy_host LANGUAGES CXX)
find_package(Threads REQUIRED)
set(RESEARCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(RADIANCE_SOURCE "${RESEARCH_ROOT}/radiance-unified-native-policy-src" CACHE PATH "Matching B2a overlaid source copy")
set(RADIANCE_HOST_BUILD "${RESEARCH_ROOT}/build-unified-native-policy-host" CACHE PATH "Matching B2a HOST Core")
if(NOT EXISTS "${RADIANCE_SOURCE}/abi/rad_model_failure_policy.h" OR NOT EXISTS "${RADIANCE_SOURCE}/core/runtime/expert_prepare.cpp")
  message(FATAL_ERROR "Requires parent-composed unified Native + permanent-policy source copy")
endif()
if(NOT EXISTS "${RADIANCE_HOST_BUILD}/core/librad_core.a")
  message(FATAL_ERROR "Parent must compile matching whole Core first")
endif()
file(STRINGS "${RADIANCE_HOST_BUILD}/CMakeCache.txt" CORE_HOME REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
string(REPLACE "CMAKE_HOME_DIRECTORY:INTERNAL=" "" CORE_HOME "${CORE_HOME}")
file(REAL_PATH "${CORE_HOME}" CORE_HOME_REAL)
file(REAL_PATH "${RADIANCE_SOURCE}" EXPECTED_CORE_REAL)
if(NOT CORE_HOME_REAL STREQUAL EXPECTED_CORE_REAL)
  message(FATAL_ERROR "Matching private Core/Program/Ctx layout required")
endif()
add_library(native_backed_source STATIC
  "${RESEARCH_ROOT}/glm53-expert-record-store-prototype/native_record_store.cpp"
  "${RESEARCH_ROOT}/glm53-record-cache-bridge-prototype/record_cache_bridge.cpp"
  "${RESEARCH_ROOT}/glm53-cache-prototype/expert_cache.cpp")
target_compile_features(native_backed_source PUBLIC cxx_std_20)
target_include_directories(native_backed_source PUBLIC
  "${RADIANCE_SOURCE}/core" "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/vendor"
  "${RADIANCE_HOST_BUILD}/rad_include"
  "${RESEARCH_ROOT}/glm53-expert-record-store-prototype"
  "${RESEARCH_ROOT}/glm53-record-cache-bridge-prototype"
  "${RESEARCH_ROOT}/glm53-cache-prototype")
target_compile_options(native_backed_source PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
add_library(unified_policy_collector MODULE collector_fixture.cpp)
target_compile_features(unified_policy_collector PRIVATE cxx_std_20)
target_include_directories(unified_policy_collector PRIVATE "${RADIANCE_SOURCE}/abi")
add_executable(ds41_unified_native_policy_host_test checker.cpp)
add_dependencies(ds41_unified_native_policy_host_test unified_policy_collector)
target_compile_definitions(ds41_unified_native_policy_host_test PRIVATE
  UNIFIED_POLICY_COLLECTOR_MODULE="$<TARGET_FILE:unified_policy_collector>")
target_compile_features(ds41_unified_native_policy_host_test PRIVATE cxx_std_20)
target_compile_options(ds41_unified_native_policy_host_test PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
target_include_directories(ds41_unified_native_policy_host_test PRIVATE "${RADIANCE_SOURCE}/arch/common")
target_link_libraries(ds41_unified_native_policy_host_test PRIVATE native_backed_source
  "-Wl,--whole-archive" "${RADIANCE_HOST_BUILD}/core/librad_core.a" "-Wl,--no-whole-archive"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_llama.a"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_httplib.a" Threads::Threads ${CMAKE_DL_LIBS})
target_link_options(ds41_unified_native_policy_host_test PRIVATE
  "LINKER:--wrap=rad_memcpy_async" "LINKER:--wrap=rad_event_record" "LINKER:--export-dynamic")
# Manual HOST-only checker; parent supplies the matching --radiance-home path.
