# ------------------------------------------------------------------------------
#  FindOpenBLAS.cmake
#
#  このスクリプトは、OpenBLAS のインクルードディレクトリとライブラリを見つけます。
#  成功した場合は以下の変数が定義されます:
#
#    OpenBLAS_FOUND          : Bool. 見つかった場合に TRUE
#    OpenBLAS_INCLUDE_DIRS   : インクルードディレクトリ一覧
#    OpenBLAS_LIBRARIES      : リンクすべきライブラリの一覧
#
#  例:
#    find_package(OpenBLAS REQUIRED)
#    if(OpenBLAS_FOUND)
#       target_include_directories(YourTarget PUBLIC ${OpenBLAS_INCLUDE_DIRS})
#       target_link_libraries(YourTarget PRIVATE ${OpenBLAS_LIBRARIES})
#    endif()
# ------------------------------------------------------------------------------

# 検索パスの定義
set(Open_BLAS_INCLUDE_SEARCH_PATHS
    /usr/include
    /usr/include/openblas
    /usr/include/openblas-base
    /usr/local/include
    /usr/local/include/openblas
    /usr/local/include/openblas-base
    /opt/OpenBLAS/include
    $ENV{OpenBLAS_HOME}
    $ENV{OpenBLAS_HOME}/include
)

set(Open_BLAS_LIB_SEARCH_PATHS
    /lib/
    /lib/openblas-base
    /lib64/
    /usr/lib
    /usr/lib/openblas-base
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /opt/OpenBLAS/lib
    $ENV{OpenBLAS}/lib
    $ENV{OpenBLAS_HOME}
    $ENV{OpenBLAS_HOME}/lib
)

# ヘッダファイルとライブラリの検索
find_path(OpenBLAS_INCLUDE_DIR
    NAMES cblas.h
    PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS}
)

find_library(OpenBLAS_LIB
    NAMES openblas
    PATHS ${Open_BLAS_LIB_SEARCH_PATHS}
)

# 変数の初期化
set(OpenBLAS_FOUND OFF)
set(OpenBLAS_INCLUDE_DIRS "")
set(OpenBLAS_LIBRARIES "")

# 見つかったかどうかを判定
if(OpenBLAS_INCLUDE_DIR AND OpenBLAS_LIB)
    set(OpenBLAS_FOUND ON)
    set(OpenBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR})
    set(OpenBLAS_LIBRARIES ${OpenBLAS_LIB})
endif()

# メッセージ出力 (必要に応じて)
if(OpenBLAS_FOUND)
    message(STATUS "Found OpenBLAS:")
    message(STATUS "  Includes: ${OpenBLAS_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${OpenBLAS_LIBRARIES}")
else()
    message(STATUS "Could NOT find OpenBLAS")
endif()

mark_as_advanced(
    OpenBLAS_INCLUDE_DIR
    OpenBLAS_LIB
)

