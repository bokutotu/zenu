/**
 * @file test_zenu_arith_cpu.cpp
 * @brief Google Test code (CPU implementations) for zenu_arith functions.
 *        This uses macros to reduce boilerplate, and tests only a few
 *        representative functions to avoid an enormous test set.
 *
 *        - Add (mat + mat)
 *        - Sub (mat - scalar_ptr)
 *        - Mul (mat_mat_assign)
 *        - Div (mat_scalar_ptr_assign)
 *
 *        We demonstrate float(f32) tests here. Similar logic can be applied
 *        for double(f64) or other functions if needed.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "zenu_compute.h"

/*=========================================================
 * ヘルパー関数: ほぼ等しいか判定
 *========================================================*/
static inline bool isNearlyEqualF32(float a, float b, float eps = 1e-5f)
{
    return (std::fabs(a - b) < eps);
}

/*=========================================================
 * テスト簡略化のためのマクロ群
 *
 * - CPUのbinary演算( mat_mat )テスト
 * - CPUのscalar_ptr演算( mat_scalar_ptr )テスト
 * - CPUのbinary複合代入( mat_mat_assign )テスト
 * - CPUのscalar_ptr複合代入( mat_scalar_ptr_assign )テスト
 *
 * テストデータはベクタで作り、OpenMPの並列などを
 * チェックしやすくするために十分に大きいサイズも良いですが、
 * ここではあまり大きすぎると時間がかかるので適度に。
 *========================================================*/

/*---------------------------------------------------------
 * マクロ: CPU用2入力演算のテスト (例: Add, Sub, etc.)
 *   FUNC: 実際に呼ぶCPU関数
 *   TEST_NAME: GoogleTest上のテスト名
 *   N: 要素数
 *   OP_DESC: 人間向けに演算内容を示す文字列("+" など)
 *--------------------------------------------------------*/
#define CPU_TEST_BIN(TEST_NAME, FUNC, N, OP_DESC)                        \
TEST(ZenuArithCpuTest, TEST_NAME)                                        \
{                                                                         \
    /* 入力データ */                                                      \
    std::vector<float> src1(N), src2(N), dst(N, 0.0f);                    \
    for (size_t i = 0; i < N; i++) {                                      \
        src1[i] = static_cast<float>(i + 1);  /* 1,2,3,... */             \
        src2[i] = static_cast<float>(0.5f * (i + 1)); /* 0.5,1.0,1.5... */\
    }                                                                     \
                                                                          \
    /* 関数呼び出し */                                                    \
    ZenuStatus st = FUNC(                                                \
        dst.data(),                                                      \
        src1.data(),                                                     \
        src2.data(),                                                     \
        /*stride_dst*/1, /*stride_src1*/1, /*stride_src2*/1,            \
        N, f32                                                          \
    );                                                                   \
    EXPECT_EQ(st, Success) << "FUNC returned error for " << OP_DESC;     \
                                                                          \
    /* 検証: dst[i] == src1[i] (OP) src2[i] */                             \
    for (size_t i = 0; i < N; i++) {                                      \
        float expectVal = 0.0f;                                          \
        /* ここではテストマクロ内で演算を行わないが、                     \
           テスト名に応じて手動で期待値を設定してもOK。 */                   \
        /* 例: add  => src1[i] + src2[i] */                               \
        expectVal = src1[i] + src2[i];                                   \
        EXPECT_TRUE(isNearlyEqualF32(dst[i], expectVal))                 \
            << "i=" << i << " " << OP_DESC << " mismatch: "              \
            << "dst=" << dst[i] << " expect=" << expectVal;              \
    }                                                                     \
}

/*---------------------------------------------------------
 * マクロ: CPU用 (mat - scalar_ptr) などスカラー演算テスト
 *   FUNC: 実際に呼ぶCPUのscalar_ptr演算関数
 *   TEST_NAME: GoogleTest上のテスト名
 *   N: 要素数
 *   OP_DESC: 演算記述("sub", "add" etc.)
 *   OP: 実際の計算式 (例: dst[i] = src[i] - c)
 *--------------------------------------------------------*/
#define CPU_TEST_SCALAR_PTR(TEST_NAME, FUNC, N, OP_DESC, OP)             \
TEST(ZenuArithCpuTest, TEST_NAME)                                        \
{                                                                         \
    std::vector<float> src(N), dst(N, 0.0f);                              \
    for (size_t i = 0; i < N; i++) {                                      \
        src[i] = static_cast<float>(i + 1);                               \
    }                                                                     \
    float scalarVal = 2.0f;                                              \
                                                                          \
    ZenuStatus st = FUNC(                                                \
        dst.data(),                                                      \
        src.data(),                                                      \
        /*stride_dst*/1, /*stride_src*/1,                                \
        &scalarVal,                                                      \
        N, f32                                                           \
    );                                                                   \
    EXPECT_EQ(st, Success) << "FUNC returned error for " << OP_DESC;     \
                                                                          \
    for (size_t i = 0; i < N; i++) {                                      \
        float expectVal = src[i] OP scalarVal;                           \
        EXPECT_TRUE(isNearlyEqualF32(dst[i], expectVal))                 \
            << "i=" << i << " " << OP_DESC << " mismatch: "              \
            << "dst=" << dst[i] << " expect=" << expectVal;              \
    }                                                                     \
}

/*---------------------------------------------------------
 * マクロ: CPU用 複合代入テスト (mat_mat_assign)
 *   例: dst[i] += src[i], dst[i] -= src[i], etc.
 *--------------------------------------------------------*/
#define CPU_TEST_ASSIGN_BIN(TEST_NAME, FUNC, N, OP_DESC, OP)             \
TEST(ZenuArithCpuTest, TEST_NAME)                                        \
{                                                                         \
    std::vector<float> dst(N), src(N);                                    \
    for (size_t i = 0; i < N; i++) {                                      \
        dst[i] = static_cast<float>(i + 1); /* 1,2,3,... */               \
        src[i] = 1.0f; /* 全要素1.0  => dst[i] <OP>= 1.0 */                \
    }                                                                     \
                                                                          \
    ZenuStatus st = FUNC(                                                \
        dst.data(),                                                      \
        src.data(),                                                      \
        /*stride_dst*/1, /*stride_src*/1,                                \
        N, f32                                                           \
    );                                                                   \
    EXPECT_EQ(st, Success) << "FUNC returned error for " << OP_DESC;     \
                                                                          \
    for (size_t i = 0; i < N; i++) {                                      \
        float expectVal = static_cast<float>(i + 1) OP 1.0f;             \
        EXPECT_TRUE(isNearlyEqualF32(dst[i], expectVal))                 \
            << "i=" << i << " " << OP_DESC << " mismatch: "              \
            << "dst=" << dst[i] << " expect=" << expectVal;              \
    }                                                                     \
}

/*---------------------------------------------------------
 * マクロ: CPU用 スカラー複合代入テスト (mat_scalar_ptr_assign)
 *   例: dst[i] /= c
 *--------------------------------------------------------*/
#define CPU_TEST_ASSIGN_SCALAR(TEST_NAME, FUNC, N, OP_DESC, OP)          \
TEST(ZenuArithCpuTest, TEST_NAME)                                        \
{                                                                         \
    std::vector<float> dst(N);                                            \
    for (size_t i = 0; i < N; i++) {                                      \
        dst[i] = static_cast<float>( (i+1)*10.0f );                       \
    }                                                                     \
    float scalarVal = 10.0f;                                             \
                                                                          \
    ZenuStatus st = FUNC(                                                \
        dst.data(),                                                      \
        /*stride_dst*/1,                                                 \
        &scalarVal,                                                      \
        N, f32                                                           \
    );                                                                   \
    EXPECT_EQ(st, Success) << "FUNC returned error for " << OP_DESC;     \
                                                                          \
    for (size_t i = 0; i < N; i++) {                                      \
        float expectVal = dst[i]; /* まだ呼び出し前の古い値 */             \
        /* ここは注意: 既に呼び出し後のdst[i]を変えてしまったので         \
           先にバックアップしておく必要がある */                           \
        /* なのでマクロの前に複製を取るか、一時変数にしておく */            \
        /* →ここではもう計算式を書いておく: oldVal / 10.0f など */         \
        /* 例として, OP_DESC="/=" =>  oldVal / c */                       \
    }                                                                     \
    /* もう一度ちゃんと計算する: */                                       \
    for (size_t i = 0; i < N; i++) {                                      \
        float oldVal = (i+1)*10.0f;                                       \
        float expectVal = oldVal OP scalarVal; /* ex: oldVal / scalarVal*/\
        EXPECT_TRUE(isNearlyEqualF32(dst[i], expectVal))                 \
            << "i=" << i << " " << OP_DESC << " mismatch: "              \
            << "dst=" << dst[i] << " expect=" << expectVal;              \
    }                                                                     \
}

/*=========================================================
 * 具体的なテスト定義
 *========================================================*/

/*-------------- 1) Add (mat + mat) --------------*/
CPU_TEST_BIN(AddMatMatCpu_F32,  /*テスト名*/
             zenu_compute_add_mat_mat_cpu, /*関数ポインタ*/
             8,   /* N=8くらい */
             "+"  /* OP_DESC */)

/*-------------- 2) Sub (mat - scalar_ptr) --------------*/
CPU_TEST_SCALAR_PTR(SubMatScalarPtrCpu_F32, /*テスト名*/
                    zenu_compute_sub_mat_scalar_ptr_cpu,
                    8, /* N=8 */
                    "-", /* OP_DESC */
                    -   /* OP (dst[i] = src[i] - c) */)

/*-------------- 3) Mul (mat_mat_assign) --------------*/
CPU_TEST_ASSIGN_BIN(MulMatMatAssignCpu_F32, /*テスト名*/
                    zenu_compute_mul_mat_mat_assign_cpu,
                    8, /*N=8*/
                    "*=", /*OP_DESC*/
                    *    /*OP*/)

/*-------------- 4) Div (mat_scalar_ptr_assign) --------------*/
CPU_TEST_ASSIGN_SCALAR(DivMatScalarPtrAssignCpu_F32, /*テスト名*/
                       zenu_compute_div_mat_scalar_ptr_assign_cpu,
                       8,  /*N=8*/
                       "/=",/*OP_DESC*/
                       /   /*OP*/)

