cmake_minimum_required(VERSION 3.24)
project(glm53_tp4_cache_prepare_lookahead_host LANGUAGES CXX)
find_package(Threads REQUIRED)
add_executable(tp4_cache_prepare_lookahead_check checker.cpp)
target_compile_features(tp4_cache_prepare_lookahead_check PRIVATE cxx_std_20)
target_compile_options(tp4_cache_prepare_lookahead_check PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(tp4_cache_prepare_lookahead_check PRIVATE Threads::Threads)
# Explicit parent-run HOST executable specification. No Core/GPU/automatic run.
