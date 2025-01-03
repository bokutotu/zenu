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
find_package(CUDAToolkit REQUIRED)  # => CUDAToolkit_FOUND で使用可否がわかる
include(${PROJECT_SOURCE_DIR}/cudnn-frontend/cmake/cuDNN.cmake)  # cuDNN 用
find_package(OpenBLAS REQUIRED)
find_package(OpenMP REQUIRED)

include(CheckCXXCompilerFlag)

add_compile_options(-march=native)

set(SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/cpu/memory.cpp
    ${PROJECT_SOURCE_DIR}/src/cpu/random.cpp
    ${PROJECT_SOURCE_DIR}/src/cpu/arithmetic.cpp
    ${PROJECT_SOURCE_DIR}/src/nvidia/memory.cpp
    ${PROJECT_SOURCE_DIR}/src/nvidia/random.cu
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
    CUDA::curand
    CUDNN::cudnn
    ${OpenBLAS_LIBRARIES}
)

#-------------------------------------------------------
# テストが有効な場合は googletest を取得・テストを作成
#   (ここで googletest_SOURCE_DIR を確保しておく)
#-------------------------------------------------------
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

    add_executable(test_all
        ${PROJECT_SOURCE_DIR}/tests/test_random.cpp
        ${PROJECT_SOURCE_DIR}/tests/test_arithmetic.cpp
    )
    target_link_libraries(test_all
        PRIVATE
        ZenuCore
        gtest_main
        ${OpenMP_CXX_LIBRARIES}
    )
    add_test(NAME test_all COMMAND test_all)
endif()

#-------------------------------------------------------
# .clangd ファイルの自動生成
#   - 2つのブロック(.cpp/.h/.cuh用, .cu用)に分けて書く
#   - テストが有効な場合は両ブロックにgoogletestのパスを追加
#-------------------------------------------------------
set(CLANGD_FILE_PATH "${PROJECT_SOURCE_DIR}/../../.clangd")
set(CLANGD_CONFIG_CONTENT "")  # .clangd 全体の内容を保持する変数

#-------------------------------------------------------
# 1) .(cpp|h|cuh) 用ブロック
#-------------------------------------------------------
string(APPEND CLANGD_CONFIG_CONTENT
"---
If:
  PathMatch: '.*\\.(cpp|h|cuh)$'
CompileFlags:
  Remove: [-xc, -xc-header]
  Add:
    - -xc++
    - -std=c++17
    - -I${PROJECT_SOURCE_DIR}/include
    - -I${PROJECT_SOURCE_DIR}/src
    - -I${PROJECT_SOURCE_DIR}/cudnn-frontend/include
")

if(CUDAToolkit_FOUND)
    string(APPEND CLANGD_CONFIG_CONTENT
"    - -I${CUDAToolkit_INCLUDE_DIRS}
")
endif()

# テストが有効 & googletest_POPULATED なら追加
if(CUDNN_FRONTEND_BUILD_TESTS)
    FetchContent_GetProperties(googletest)
    if(googletest_POPULATED)
        string(APPEND CLANGD_CONFIG_CONTENT
"    - -I${googletest_SOURCE_DIR}/googletest/include
")
    endif()
endif()

# ブロック終了: 改行入れておく
string(APPEND CLANGD_CONFIG_CONTENT "\n")

#-------------------------------------------------------
# 2) .cu 用ブロック (CUDA モード)
#-------------------------------------------------------
string(APPEND CLANGD_CONFIG_CONTENT
"---
If:
  PathMatch: '.*\\.cu$'
CompileFlags:
  Remove: [-xc, -xc-header]
  Add:
    - -x
    - cuda
    - -std=c++17
    - -I${PROJECT_SOURCE_DIR}/include
    - -I${PROJECT_SOURCE_DIR}/src
    - -I${PROJECT_SOURCE_DIR}/cudnn-frontend/include
")

if(CUDAToolkit_FOUND)
    string(APPEND CLANGD_CONFIG_CONTENT
"    - -I${CUDAToolkit_INCLUDE_DIRS}
")
endif()

# テストが有効 & googletest_POPULATED なら追加
if(CUDNN_FRONTEND_BUILD_TESTS)
    FetchContent_GetProperties(googletest)
    if(googletest_POPULATED)
        string(APPEND CLANGD_CONFIG_CONTENT
"    - -I${googletest_SOURCE_DIR}/googletest/include
")
    endif()
endif()

# ブロック終了: 改行入れておく
string(APPEND CLANGD_CONFIG_CONTENT "\n")

#-------------------------------------------------------
# 最後にファイルに出力
#-------------------------------------------------------
file(WRITE "${CLANGD_FILE_PATH}" "${CLANGD_CONFIG_CONTENT}")

#-------------------------------------------------------
# ライブラリのインストール設定
#-------------------------------------------------------
install(TARGETS ZenuCore
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

