cmake_minimum_required(VERSION 3.24)
project(glm53_synthetic_sharded_native_provider LANGUAGES CXX)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "Candidate retains Linux S1 NOREPLACE publication and POSIX metadata")
endif()
find_package(Threads REQUIRED)
set(RESEARCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(RADIANCE_SOURCE "${RESEARCH_ROOT}/radiance-src" CACHE PATH "Pinned read-only matching Core source")
set(RADIANCE_HOST_BUILD "${RADIANCE_SOURCE}/build-iq2-host" CACHE PATH "Parent-selected matching existing HOST Core")
if(NOT EXISTS "${RADIANCE_HOST_BUILD}/core/librad_core.a")
  message(FATAL_ERROR "Pass matching existing HOST Core archives; no implicit Core build")
endif()
add_library(glm53_sharded_native_store STATIC candidate_native_record_store.cpp rank_native_store.cpp)
target_compile_features(glm53_sharded_native_store PUBLIC cxx_std_20)
target_include_directories(glm53_sharded_native_store PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
  "${RADIANCE_SOURCE}/core" "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/vendor"
  "${RADIANCE_HOST_BUILD}/rad_include")
target_compile_options(glm53_sharded_native_store PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
target_link_libraries(glm53_sharded_native_store PUBLIC Threads::Threads)
add_executable(glm53_sharded_native_store_test sharded_native_store_test.cpp)
target_compile_features(glm53_sharded_native_store_test PRIVATE cxx_std_20)
target_compile_options(glm53_sharded_native_store_test PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
target_include_directories(glm53_sharded_native_store_test PRIVATE "${RADIANCE_SOURCE}/tools")
target_link_libraries(glm53_sharded_native_store_test PRIVATE glm53_sharded_native_store
  "${RADIANCE_HOST_BUILD}/core/librad_core.a"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_llama.a"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_httplib.a" dl Threads::Threads)
target_link_options(glm53_sharded_native_store_test PRIVATE "LINKER:--wrap=pread")
# Parent manually invokes only this CPU test; no Core adapter/serving/install/device target.
