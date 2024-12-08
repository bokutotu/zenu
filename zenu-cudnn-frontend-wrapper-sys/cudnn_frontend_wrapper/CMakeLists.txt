cmake_minimum_required(VERSION 3.18)
project(cudnn_frontend_wrapper VERSION 1.8.0)

option(CUDNN_FRONTEND_SKIP_JSON_LIB "Defines whether FE should not include nlohmann/json.hpp." OFF)
option(CUDNN_FRONTEND_BUILD_TESTS "Build tests for cudnn_frontend_wrapper" OFF)

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

# C++17 対応
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# cuDNNをCUDNN::cudnnターゲットとして読み込み
include(${PROJECT_SOURCE_DIR}/cudnn-frontend/cmake/cuDNN.cmake)

set(SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/batchnorm.cpp
    ${PROJECT_SOURCE_DIR}/src/conv.cpp
    ${PROJECT_SOURCE_DIR}/src/cudnn_frontend_wrapper.cpp
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
)

# 静的ライブラリ作成
add_library(cudnn_frontend_wrapper STATIC ${SRC_FILES})

# コンパイル定義フラグをターゲットへ設定
target_compile_definitions(cudnn_frontend_wrapper PUBLIC
    $<$<BOOL:${CUDNN_FRONTEND_SKIP_JSON_LIB}>:CUDNN_FRONTEND_SKIP_JSON_LIB>
)

# インクルードディレクトリ設定
target_include_directories(cudnn_frontend_wrapper PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/cudnn-frontend/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# デバッグビルドでは-gオプションを付与
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cudnn_frontend_wrapper PUBLIC -g)
endif()

# リンクライブラリ設定
target_link_libraries(cudnn_frontend_wrapper
    Threads::Threads
    CUDA::cudart
    CUDA::cuda_driver
    CUDNN::cudnn
)

if(CUDNN_FRONTEND_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v2.13.7
    )
    FetchContent_MakeAvailable(Catch2)

    set(TEST_SRC_FILES
        ${PROJECT_SOURCE_DIR}/tests/batchnorm.cpp
        ${PROJECT_SOURCE_DIR}/tests/conv.cpp
    )

    add_executable(run_tests ${TEST_SRC_FILES})

    target_link_libraries(run_tests
        cudnn_frontend_wrapper
        Catch2::Catch2
    )

    # テスト登録
    include(CTest)
    add_test(NAME run_tests COMMAND run_tests)
endif()

# ccls用の設定ファイル出力（任意）
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/.ccls "
%compile_commands.json=build/compile_commands.json
%clang
-std=c++17
-I${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include
")

install(TARGETS cudnn_frontend_wrapper
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
