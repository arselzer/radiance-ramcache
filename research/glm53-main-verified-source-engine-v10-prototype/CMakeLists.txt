cmake_minimum_required(VERSION 3.24)
project(glm53_main_verified_source_engine LANGUAGES CXX)
find_package(Threads REQUIRED)
set(RESEARCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(RADIANCE_SOURCE "${RESEARCH_ROOT}/radiance-glm-main-verified-source-v8-src" CACHE PATH "Exact frozen foundation Core")
set(RADIANCE_HOST_BUILD "${RESEARCH_ROOT}/build-glm53-main-verified-engine-core-v8-host-release" CACHE PATH "Parent-built matching whole HOST Core")
if(NOT EXISTS "${RADIANCE_HOST_BUILD}/core/librad_core.a")
  message(FATAL_ERROR "Parent must supply matching whole Core build; no implicit Core build")
endif()
file(STRINGS "${RADIANCE_HOST_BUILD}/CMakeCache.txt" CORE_HOME REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
string(REPLACE "CMAKE_HOME_DIRECTORY:INTERNAL=" "" CORE_HOME "${CORE_HOME}")
file(REAL_PATH "${CORE_HOME}" ACTUAL_HOME)
file(REAL_PATH "${RADIANCE_SOURCE}" EXPECTED_HOME)
if(NOT ACTUAL_HOME STREQUAL EXPECTED_HOME)
  message(FATAL_ERROR "Private quote headers/archive source mismatch")
endif()
option(RADIANCE_AQL_CORE "Explicit matching HIP/AQL Core build (no execution)" OFF)
file(STRINGS "${RADIANCE_HOST_BUILD}/CMakeCache.txt" HOST_BACKEND REGEX "^RAD_WITH_HIP:STRING=OFF$")
if(HOST_BACKEND AND RADIANCE_AQL_CORE)
  message(FATAL_ERROR "AQL fixture profile cannot consume HOST Core")
elseif(NOT HOST_BACKEND AND NOT RADIANCE_AQL_CORE)
  message(FATAL_ERROR "Explicit -DRADIANCE_AQL_CORE=ON required for a non-HOST Core")
endif()
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/CORE_SOURCE_PINS.json" PINS)
string(JSON COUNT LENGTH "${PINS}")
math(EXPR LAST "${COUNT}-1")
foreach(I RANGE 0 ${LAST})
  string(JSON REL MEMBER "${PINS}" ${I})
  string(JSON EXPECTED GET "${PINS}" "${REL}")
  file(SHA256 "${RADIANCE_SOURCE}/${REL}" ACTUAL)
  if(NOT ACTUAL STREQUAL EXPECTED)
    message(FATAL_ERROR "Frozen foundation source mismatch: ${REL}")
  endif()
endforeach()
add_executable(glm53_main_verified_engine_manual manual_engine.cpp)
target_compile_features(glm53_main_verified_engine_manual PRIVATE cxx_std_20)
target_include_directories(glm53_main_verified_engine_manual PRIVATE "${RADIANCE_SOURCE}/core" "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/vendor")
target_link_libraries(glm53_main_verified_engine_manual PRIVATE
  "-Wl,--whole-archive" "${RADIANCE_HOST_BUILD}/core/librad_core.a" "-Wl,--no-whole-archive"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_llama.a" "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_httplib.a" Threads::Threads ${CMAKE_DL_LIBS})
target_link_options(glm53_main_verified_engine_manual PRIVATE "LINKER:--export-dynamic")
if(RADIANCE_AQL_CORE)
  find_library(GLM_HIP amdhip64 REQUIRED)
  find_library(GLM_HSA hsa-runtime64 REQUIRED)
  target_compile_definitions(glm53_main_verified_engine_manual PRIVATE RAD_HAVE_HIP=1)
  target_link_libraries(glm53_main_verified_engine_manual PRIVATE ${GLM_HIP} ${GLM_HSA})
endif()
if(HOST_BACKEND)
# CPU-only actual header/tokenizer preflight and explicit synthetic HOST mask proof.
add_executable(glm53_main_vocabulary_check vocabulary_checker.cpp)
target_compile_features(glm53_main_vocabulary_check PRIVATE cxx_std_20)
target_compile_options(glm53_main_vocabulary_check PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
target_include_directories(glm53_main_vocabulary_check PRIVATE "${RADIANCE_SOURCE}/core" "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/vendor")
target_link_libraries(glm53_main_vocabulary_check PRIVATE
  "-Wl,--whole-archive" "${RADIANCE_HOST_BUILD}/core/librad_core.a" "-Wl,--no-whole-archive"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_llama.a" "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_httplib.a" Threads::Threads ${CMAKE_DL_LIBS})
target_link_options(glm53_main_vocabulary_check PRIVATE "LINKER:--export-dynamic"
  "LINKER:--wrap=rad_dev_count" "LINKER:--wrap=rad_dev_set" "LINKER:--wrap=rad_dev_props"
  "LINKER:--wrap=rad_dev_alloc" "LINKER:--wrap=rad_stream_create")
add_executable(glm53_main_verified_engine_gate gate_checker.cpp)
target_compile_features(glm53_main_verified_engine_gate PRIVATE cxx_std_20)
target_include_directories(glm53_main_verified_engine_gate PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${RADIANCE_SOURCE}/core" "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/vendor")
target_link_libraries(glm53_main_verified_engine_gate PRIVATE
  "-Wl,--whole-archive" "${RADIANCE_HOST_BUILD}/core/librad_core.a" "-Wl,--no-whole-archive"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_llama.a" "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_httplib.a" Threads::Threads ${CMAKE_DL_LIBS})
target_link_options(glm53_main_verified_engine_gate PRIVATE "LINKER:--export-dynamic" "LINKER:--wrap=open" "LINKER:--wrap=pread" "LINKER:--wrap=rad_dev_alloc")
add_executable(glm53_main_bounded_decode_check decode_checker.cpp)
target_compile_features(glm53_main_bounded_decode_check PRIVATE cxx_std_20)
target_include_directories(glm53_main_bounded_decode_check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${RADIANCE_SOURCE}/core" "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/vendor")
target_link_libraries(glm53_main_bounded_decode_check PRIVATE
  "-Wl,--whole-archive" "${RADIANCE_HOST_BUILD}/core/librad_core.a" "-Wl,--no-whole-archive"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_llama.a" "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_httplib.a" Threads::Threads ${CMAKE_DL_LIBS})
target_link_options(glm53_main_bounded_decode_check PRIVATE "LINKER:--export-dynamic" "LINKER:--wrap=open" "LINKER:--wrap=pread" "LINKER:--wrap=rad_dev_alloc")
add_executable(tiny_native_foundation_check foundation_checker.cpp)
target_compile_features(tiny_native_foundation_check PRIVATE cxx_std_20)
target_include_directories(tiny_native_foundation_check PRIVATE "${RADIANCE_SOURCE}/core" "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/vendor")
target_compile_options(tiny_native_foundation_check PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
target_link_libraries(tiny_native_foundation_check PRIVATE
  "-Wl,--whole-archive" "${RADIANCE_HOST_BUILD}/core/librad_core.a" "-Wl,--no-whole-archive"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_llama.a" "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_httplib.a" Threads::Threads ${CMAKE_DL_LIBS})
target_link_options(tiny_native_foundation_check PRIVATE "LINKER:--export-dynamic" "LINKER:--wrap=rad_dev_alloc" "LINKER:--wrap=rad_dev_free" "LINKER:--wrap=rad_memcpy_async" "LINKER:--wrap=rad_event_record" "LINKER:--wrap=rad_event_sync")
add_subdirectory(contract)
option(FOUNDATION_WITH_HOST_REGRESSIONS "Build unchanged matching HOST regressions" ON)
if(FOUNDATION_WITH_HOST_REGRESSIONS)
  add_subdirectory(regressions/compact)
endif()
# No CTest, automatic execution, installation or service target.
# The explicit manual target below invokes actual Engine.start when the parent runs it.

add_executable(tiny_engine_native_transport_check checker.cpp)
target_compile_features(tiny_engine_native_transport_check PRIVATE cxx_std_20)
target_compile_options(tiny_engine_native_transport_check PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
target_include_directories(tiny_engine_native_transport_check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/contract" "${RADIANCE_SOURCE}/core" "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/vendor" "${RADIANCE_SOURCE}")
target_link_libraries(tiny_engine_native_transport_check PRIVATE
  "-Wl,--whole-archive" "${RADIANCE_HOST_BUILD}/core/librad_core.a" "-Wl,--no-whole-archive"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_llama.a" "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_httplib.a" Threads::Threads ${CMAKE_DL_LIBS})
# Original (un-isolated) fixture writer is a caller/test-data owner. Engine's
# fixed factory copies live in mechanically isolated Core namespaces.
target_sources(tiny_engine_native_transport_check PRIVATE contract/fixture_writer.cpp contract/profile.cpp contract/shape_profile.cpp)
target_link_options(tiny_engine_native_transport_check PRIVATE "LINKER:--export-dynamic" "LINKER:--wrap=rad_dev_alloc" "LINKER:--wrap=rad_dev_free" "LINKER:--wrap=rad_memcpy_async" "LINKER:--wrap=open" "LINKER:--wrap=pread")
add_dependencies(tiny_engine_native_transport_check glm53_tiny_startup_arch)

# These new targets execute only with an explicit parent invocation. No CTest,
# install or serving target is added; the experimental architecture remains OFF.
add_executable(glm53_tp4_cache_session_check session_checker.cpp)
target_compile_features(glm53_tp4_cache_session_check PRIVATE cxx_std_20)
target_compile_options(glm53_tp4_cache_session_check PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
target_include_directories(glm53_tp4_cache_session_check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${RADIANCE_SOURCE}/core" "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/vendor")
target_link_libraries(glm53_tp4_cache_session_check PRIVATE
  "-Wl,--whole-archive" "${RADIANCE_HOST_BUILD}/core/librad_core.a" "-Wl,--no-whole-archive"
  "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_llama.a" "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_httplib.a" Threads::Threads ${CMAKE_DL_LIBS})
target_link_options(glm53_tp4_cache_session_check PRIVATE "LINKER:--export-dynamic")
if(FOUNDATION_WITH_HOST_REGRESSIONS)
  add_executable(glm53_tp4_mover_fence_check mover_wait_checker.cpp)
  target_compile_features(glm53_tp4_mover_fence_check PRIVATE cxx_std_20)
  target_compile_options(glm53_tp4_mover_fence_check PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
  target_include_directories(glm53_tp4_mover_fence_check PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${RADIANCE_SOURCE}/arch/common" "${RADIANCE_SOURCE}/tools")
  target_link_libraries(glm53_tp4_mover_fence_check PRIVATE p3_native_source
    "-Wl,--whole-archive" "${RADIANCE_HOST_BUILD}/core/librad_core.a" "-Wl,--no-whole-archive"
    "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_llama.a" "${RADIANCE_HOST_BUILD}/vendor/librad_vendor_httplib.a" Threads::Threads ${CMAKE_DL_LIBS})
  target_link_options(glm53_tp4_mover_fence_check PRIVATE "LINKER:--export-dynamic" "LINKER:--wrap=rad_memcpy_async" "LINKER:--wrap=rad_event_record" "LINKER:--wrap=rad_stream_sync")
endif()

endif() # unchanged HOST bodies + explicit new HOST gate, never auto-run
