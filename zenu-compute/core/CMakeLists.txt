cmake_minimum_required(VERSION 3.18)
project(ZenuCore VERSION 1.8.0)

# --- CMAKE_MODULE_PATH の設定 (自作モジュールを置いている場所) ---
# "cmake/Modules" の部分は実際のディレクトリに合わせてください
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

option(CUDNN_FRONTEND_SKIP_JSON_LIB "Defines whether FE should not include nlohmann/json.hpp." OFF)
option(CUDNN_FRONTEND_BUILD_TESTS "Build tests for ZenuCore" OFF)

# ---- C++17 対応 ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- CUDA を使う場合は言語を有効化しておく ----
enable_language(CUDA)

# ---- スレッドライブラリの検出 ----
find_package(Threads REQUIRED)

# ---- CUDA Toolkit の検出 ----
find_package(CUDAToolkit REQUIRED)

# ---- cuDNNをCUDNN::cudnnターゲットとして読み込み ----
include(${PROJECT_SOURCE_DIR}/cudnn-frontend/cmake/cuDNN.cmake)

# ---- OpenBLAS 探索 (REQUIRED にすると見つからなかった場合ビルドエラー) ----
find_package(OpenBLAS REQUIRED)

# ---- ソースファイル ----
set(SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/conv.cpp
)

# ---- 静的ライブラリ作成 ----
add_library(ZenuCore STATIC ${SRC_FILES})

# ---- コンパイル定義フラグをターゲットへ設定 ----
target_compile_definitions(ZenuCore PUBLIC
    $<$<BOOL:${CUDNN_FRONTEND_SKIP_JSON_LIB}>:CUDNN_FRONTEND_SKIP_JSON_LIB>
)

# ---- インクルードディレクトリ設定 ----
target_include_directories(ZenuCore PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/cudnn-frontend/include
    ${CUDAToolkit_INCLUDE_DIRS}

    # OpenBLAS が見つかった場合にインクルードディレクトリを追加
    ${OpenBLAS_INCLUDE_DIRS}
)

# ---- デバッグビルドでは-gオプションを付与 ----
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(ZenuCore PUBLIC -g)
endif()

# ---- リンクライブラリ設定 ----
target_link_libraries(ZenuCore
    PRIVATE
    Threads::Threads
    CUDA::cudart         # または CUDAToolkit::cudart 等、環境に合わせる
    CUDA::cuda_driver    # または CUDAToolkit::nvrtc 等
    CUDNN::cudnn

    # OpenBLAS が見つかった場合にライブラリを追加
    ${OpenBLAS_LIBRARIES}
)

###
# ccls 用の設定ファイル出力（例）
###
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/../../.ccls "
%compile_commands.json=build/compile_commands.json
%clang
-std=c++17
-I${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include
")

###
# テストターゲットの追加 (option で ON/OFF)
###
if(CUDNN_FRONTEND_BUILD_TESTS)
    enable_testing()

    # 例: conv のテストを行う実行ファイル
    add_executable(test_ZenuCore
        ${PROJECT_SOURCE_DIR}/test/test_conv.cpp
    )
    target_link_libraries(test_ZenuCore PRIVATE ZenuCore)
    
    # ctest で実行できるようテスト登録
    add_test(NAME TestCudnnFrontendWrapper COMMAND test_ZenuCore)
endif()

# ---- インストール設定 ----
install(TARGETS ZenuCore
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

