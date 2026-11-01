cmake_minimum_required(VERSION 3.20)
project(moss_llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_compile_options(/utf-8 /W3 /EHsc /fp:fast /D_CRT_SECURE_NO_WARNINGS)
  add_compile_definitions(_WIN32_WINNT=0x0A00 NOMINMAX WIN32_LEAN_AND_MEAN)
  add_compile_options(/arch:SSE2)   # 基线段：不要求 AVX2
endif()

set(MOSS_CORE_SOURCES
  src/fp16.cpp
  src/cpu_features.cpp
  src/quant.cpp
  src/gemv.cpp
  src/gemv_avx2.cpp
)

add_library(moss_core STATIC ${MOSS_CORE_SOURCES})
target_include_directories(moss_core PUBLIC include)

if(MSVC)
  # AVX2 只对这一个文件开，其余保持 SSE2 基线，运行时再分派
  set_source_files_properties(src/gemv_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
endif()

add_executable(test_kernels tests/test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE moss_core)
