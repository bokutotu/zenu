cmake_minimum_required(VERSION 3.18)
project(ZenuCore VERSION 1.8.0)

# --- CMAKE_MODULE_PATH の設定 (自作モジュールを置いている場所) ---
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

option(CUDNN_FRONTEND_SKIP_JSON_LIB "Defines whether FE should not include nlohmann/json.hpp." OFF)
option(CUDNN_FRONTEND_BUILD_TESTS "Build tests for ZenuCore" OFF)

# ---- C++17 対応 ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- CUDA を使う場合は言語を有効化しておく ----
enable_language(CUDA)

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
include(${PROJECT_SOURCE_DIR}/cudnn-frontend/cmake/cuDNN.cmake)
find_package(OpenBLAS REQUIRED)
find_package(OpenMP REQUIRED)

include(CheckCXXCompilerFlag)

# AVX 判定
check_cxx_compiler_flag("-mavx" HAS_AVX)
if(HAS_AVX)
    message(STATUS "AVX is supported")
    add_compile_options(-mavx)
endif()

# AVX2 判定
check_cxx_compiler_flag("-mavx2" HAS_AVX2)
if(HAS_AVX2)
    message(STATUS "AVX2 is supported")
    add_compile_options(-mavx2)
endif()

# AVX512 判定
check_cxx_compiler_flag("-mavx512f" HAS_AVX512)
if(HAS_AVX512)
    message(STATUS "AVX512 is supported")
    add_compile_options(-mavx512f)
endif()

# FMA 判定
check_cxx_compiler_flag("-mfma" HAS_FMA)
if(HAS_FMA)
    message(STATUS "FMA is supported")
    add_compile_options(-mfma)
endif()

# NEON 判定 (ARM)
check_cxx_compiler_flag("-mfpu=neon" HAS_NEON)
if(HAS_NEON)
    message(STATUS "NEON is supported")
    add_compile_options(-mfpu=neon)
endif()

set(SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/cpu/memory.cpp
    ${PROJECT_SOURCE_DIR}/src/cpu/random.cpp
    ${PROJECT_SOURCE_DIR}/src/nvidia/memory.cpp
    ${PROJECT_SOURCE_DIR}/src/nvidia/random.cpp
)

add_library(ZenuCore STATIC ${SRC_FILES})

if(OpenMP_CXX_FOUND)
    target_compile_options(ZenuCore PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(ZenuCore PUBLIC ${OpenMP_CXX_LIBRARIES})
endif()

target_compile_definitions(ZenuCore PUBLIC
    $<$<BOOL:${CUDNN_FRONTEND_SKIP_JSON_LIB}>:CUDNN_FRONTEND_SKIP_JSON_LIB>
)

target_include_directories(ZenuCore PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/cudnn-frontend/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenBLAS_INCLUDE_DIRS}
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(ZenuCore PUBLIC -g)
endif()

target_link_libraries(ZenuCore
    PRIVATE
    Threads::Threads
    CUDA::cudart
    CUDA::cuda_driver
    CUDNN::cudnn
    ${OpenBLAS_LIBRARIES}
)

file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/../../.ccls "
%compile_commands.json=build/compile_commands.json
%clang
-std=c++17
-I${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include
")

if(CUDNN_FRONTEND_BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(test_random
        ${PROJECT_SOURCE_DIR}/tests/test_random.cpp
    )
    target_link_libraries(test_random PRIVATE ZenuCore gtest_main ${OpenMP_CXX_LIBRARIES})
    add_test(NAME test_random COMMAND test_random)
endif()

install(TARGETS ZenuCore
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

