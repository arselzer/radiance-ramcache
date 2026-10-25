cmake_minimum_required(VERSION 3.24)
project(glm53_synthetic_record_cache_bridge LANGUAGES CXX)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The frozen native store requires Linux NOREPLACE publication")
endif()
find_package(Threads REQUIRED)
set(RESEARCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(RADIANCE_SOURCE "${RESEARCH_ROOT}/radiance-src" CACHE PATH "Pinned read-only source")
set(RADIANCE_HOST_BUILD "${RADIANCE_SOURCE}/build-iq2-host" CACHE PATH "Matching existing HOST Core; no implicit build")
if(NOT EXISTS "${RADIANCE_HOST_BUILD}/core/librad_core.a")
  message(FATAL_ERROR "Pass matching existing HOST Core archives")
endif()
# Compile dependencies from frozen source without modifying them, allowing matching sanitizer
# instrumentation throughout this standalone adapter. No other prototype CMake is invoked.
add_library(glm53_record_cache_bridge STATIC record_cache_bridge.cpp
  "${RESEARCH_ROOT}/glm53-expert-record-store-prototype/native_record_store.cpp"
  "${RESEARCH_ROOT}/glm53-cache-prototype/expert_cache.cpp")
target_compile_features(glm53_record_cache_bridge PUBLIC cxx_std_20)
target_include_directories(glm53_record_cache_bridge PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
  "${RESEARCH_ROOT}/glm53-expert-record-store-prototype" "${RESEARCH_ROOT}/glm53-cache-prototype"
  "${RADIANCE_SOURCE}/core" "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/vendor"
  "${RADIANCE_HOST_BUILD}/rad_include")
target_compile_options(glm53_record_cache_bridge PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
target_link_libraries(glm53_record_cache_bridge PUBLIC Threads::Threads)
add_executable(glm53_record_cache_bridge_test record_cache_bridge_test.cpp)
target_compile_features(glm53_record_cache_bridge_test PRIVATE cxx_std_20)
target_compile_options(glm53_record_cache_bridge_test PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
target_include_directories(glm53_record_cache_bridge_test PRIVATE "${RADIANCE_SOURCE}/tools")
target_link_libraries(glm53_record_cache_bridge_test PRIVATE glm53_record_cache_bridge
  "${RADIANCE_HOST_BUILD}/core/librad_core.a"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_llama.a"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_httplib.a" dl Threads::Threads)
# CPU-only manual test; no device/backend queries, serving scan, installation or full weights.
