cmake_minimum_required(VERSION 3.24)
project(glm53_real_main_record_packer LANGUAGES CXX)
find_package(Threads REQUIRED)
set(RESEARCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(RADIANCE_SOURCE "${RESEARCH_ROOT}/radiance-src" CACHE PATH "Exact original frozen Core source")
set(RADIANCE_BUILD "${RESEARCH_ROOT}/build-ds41-aql" CACHE PATH "Matching parent-built whole AQL Core")
if(NOT EXISTS "${RADIANCE_BUILD}/core/librad_core.a")
  message(FATAL_ERROR "Supply matching existing whole AQL Core; no implicit Core build")
endif()
file(STRINGS "${RADIANCE_BUILD}/CMakeCache.txt" CORE_HOME REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
string(REPLACE "CMAKE_HOME_DIRECTORY:INTERNAL=" "" CORE_HOME "${CORE_HOME}")
file(REAL_PATH "${CORE_HOME}" ACTUAL_HOME)
file(REAL_PATH "${RADIANCE_SOURCE}" EXPECTED_HOME)
if(NOT ACTUAL_HOME STREQUAL EXPECTED_HOME)
  message(FATAL_ERROR "Actual private Core source/header/archive mismatch")
endif()
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/CORE_SOURCE_PINS.json" PINS)
string(JSON COUNT LENGTH "${PINS}")
math(EXPR LAST "${COUNT}-1")
foreach(I RANGE 0 ${LAST})
  string(JSON REL MEMBER "${PINS}" ${I})
  string(JSON EXPECTED GET "${PINS}" "${REL}")
  file(SHA256 "${RADIANCE_SOURCE}/${REL}" ACTUAL)
  if(NOT ACTUAL STREQUAL EXPECTED)
    message(FATAL_ERROR "Pinned metadata planning source changed: ${REL}")
  endif()
endforeach()
find_library(PLAN_HIP_LIBRARY amdhip64 REQUIRED)
find_library(PLAN_HSA_LIBRARY hsa-runtime64 REQUIRED)
add_executable(glm53_real_main_record_check sample_packer.cpp main_records.cpp geometry/main_metadata.cpp)
target_compile_features(glm53_real_main_record_check PRIVATE cxx_std_20)
target_compile_definitions(glm53_real_main_record_check PRIVATE RAD_HAVE_HIP=1)
target_compile_options(glm53_real_main_record_check PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
target_include_directories(glm53_real_main_record_check PRIVATE "${RADIANCE_SOURCE}" "${RADIANCE_SOURCE}/core"
  "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/vendor" "${RADIANCE_BUILD}/rad_include")
target_link_libraries(glm53_real_main_record_check PRIVATE
  "-Wl,--whole-archive" "${RADIANCE_BUILD}/core/librad_core.a" "-Wl,--no-whole-archive"
  "${RADIANCE_BUILD}/vendor/librad_vendor_llama.a" "${RADIANCE_BUILD}/vendor/librad_vendor_httplib.a"
  ${PLAN_HIP_LIBRARY} ${PLAN_HSA_LIBRARY} Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(glm53_real_main_record_check PROPERTIES ENABLE_EXPORTS ON)
# No CTest, Core rebuild, model allocation, queue/launch target, install or serving change.

target_link_options(glm53_real_main_record_check PRIVATE "LINKER:--wrap=pread" "LINKER:--wrap=pwrite" "LINKER:--wrap=renameat2" "LINKER:--wrap=fstat")

# bundle_records.cpp includes the byte-identical v3 main_records.cpp primitives;
# never list main_records.cpp a second time in this target.
add_executable(glm53_main_store_assembler assemble_store.cpp bundle_records.cpp geometry/main_metadata.cpp)
target_compile_features(glm53_main_store_assembler PRIVATE cxx_std_20)
target_compile_definitions(glm53_main_store_assembler PRIVATE RAD_HAVE_HIP=1)
target_compile_options(glm53_main_store_assembler PRIVATE -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off)
target_include_directories(glm53_main_store_assembler PRIVATE "${RADIANCE_SOURCE}" "${RADIANCE_SOURCE}/core"
  "${RADIANCE_SOURCE}/abi" "${RADIANCE_SOURCE}/vendor" "${RADIANCE_BUILD}/rad_include")
target_link_libraries(glm53_main_store_assembler PRIVATE
  "-Wl,--whole-archive" "${RADIANCE_BUILD}/core/librad_core.a" "-Wl,--no-whole-archive"
  "${RADIANCE_BUILD}/vendor/librad_vendor_llama.a" "${RADIANCE_BUILD}/vendor/librad_vendor_httplib.a"
  ${PLAN_HIP_LIBRARY} ${PLAN_HSA_LIBRARY} Threads::Threads ${CMAKE_DL_LIBS})
target_link_options(glm53_main_store_assembler PRIVATE "LINKER:--wrap=pread" "LINKER:--wrap=pwrite" "LINKER:--wrap=renameat2" "LINKER:--wrap=fstat")
set_target_properties(glm53_main_store_assembler PROPERTIES ENABLE_EXPORTS ON)
