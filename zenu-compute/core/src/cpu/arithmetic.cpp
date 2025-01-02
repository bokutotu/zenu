#include "zenu_compute.h"
#include "iter.h"

#if defined(__AVX512F__)
#  include <immintrin.h>
#elif defined(__AVX2__)
#  include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#else
#  define SIMD_NONE
#endif

//---------------------------------------------------
// (1) スカラー演算子をマクロ化
//---------------------------------------------------
#define OP_ADD +
#define OP_SUB -
#define OP_MUL *
#define OP_DIV /

//---------------------------------------------------
// (2) まず環境ごとにマクロ再定義
//     float32 用 (DEF_VEC_F32_OP)
//---------------------------------------------------
#if defined(__AVX512F__)

// AVX512F の場合
#define DEF_VEC_F32_OP(NAME, OPMACRO, AVX512_OP, AVX2_OP, NEON_OP)       \
    static inline __m512 NAME##_vec_f32(__m512 a, __m512 b) {           \
        return _mm512_##AVX512_OP##_ps(a, b);                           \
    }

#elif defined(__AVX2__)

// AVX2 の場合
#define DEF_VEC_F32_OP(NAME, OPMACRO, AVX512_OP, AVX2_OP, NEON_OP)       \
    static inline __m256 NAME##_vec_f32(__m256 a, __m256 b) {           \
        return _mm256_##AVX2_OP##_ps(a, b);                             \
    }

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)

// NEON の場合
#define DEF_VEC_F32_OP(NAME, OPMACRO, AVX512_OP, AVX2_OP, NEON_OP)       \
    static inline float32x4_t NAME##_vec_f32(float32x4_t a, float32x4_t b) { \
        return v##NEON_OP##q_f32(a, b);                                  \
    }

#else

// どれでもない場合 (fallback)
#define DEF_VEC_F32_OP(NAME, OPMACRO, AVX512_OP, AVX2_OP, NEON_OP)       \
    static inline float NAME##_vec_f32(float a, float b) {              \
        return a OPMACRO b;                                             \
    }

#endif


//---------------------------------------------------
// (3) float64 用 (DEF_VEC_F64_OP)
//     こちらも同様に、環境別にマクロ再定義
//---------------------------------------------------
#if defined(__AVX512F__)

#define DEF_VEC_F64_OP(NAME, OPMACRO, AVX512_OP, AVX2_OP, NEON_OP)       \
    static inline __m512d NAME##_vec_f64(__m512d a, __m512d b) {         \
        return _mm512_##AVX512_OP##_pd(a, b);                           \
    }

#elif defined(__AVX2__)

#define DEF_VEC_F64_OP(NAME, OPMACRO, AVX512_OP, AVX2_OP, NEON_OP)       \
    static inline __m256d NAME##_vec_f64(__m256d a, __m256d b) {         \
        return _mm256_##AVX2_OP##_pd(a, b);                             \
    }

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)

#define DEF_VEC_F64_OP(NAME, OPMACRO, AVX512_OP, AVX2_OP, NEON_OP)       \
    static inline float64x2_t NAME##_vec_f64(float64x2_t a, float64x2_t b) { \
        return v##NEON_OP##q_f64(a, b);                                  \
    }

#else

#define DEF_VEC_F64_OP(NAME, OPMACRO, AVX512_OP, AVX2_OP, NEON_OP)       \
    static inline double NAME##_vec_f64(double a, double b) {           \
        return a OPMACRO b;                                             \
    }

#endif


//---------------------------------------------------
// (4) float32_strN 用 (DEF_VEC_F32_OP_STRN)
//     環境別にマクロ再定義
//---------------------------------------------------
#if defined(__AVX512F__)

#define DEF_VEC_F32_OP_STRN(NAME, OPMACRO, AVX512_OP, AVX2_OP)           \
    static inline __m512 NAME##_vec_f32_strN(__m512 a, __m512 b) {       \
        return _mm512_##AVX512_OP##_ps(a, b);                            \
    }

#elif defined(__AVX2__)

#define DEF_VEC_F32_OP_STRN(NAME, OPMACRO, AVX512_OP, AVX2_OP)           \
    static inline __m256 NAME##_vec_f32_strN(__m256 a, __m256 b) {       \
        return _mm256_##AVX2_OP##_ps(a, b);                              \
    }

#else

#define DEF_VEC_F32_OP_STRN(NAME, OPMACRO, AVX512_OP, AVX2_OP)           \
    static inline float NAME##_vec_f32_strN(float a, float b) {          \
        return a OPMACRO b;                                              \
    }

#endif


//---------------------------------------------------
// (5) float64_strN 用 (DEF_VEC_F64_OP_STRN)
//     環境別にマクロ再定義
//---------------------------------------------------
#if defined(__AVX512F__)

#define DEF_VEC_F64_OP_STRN(NAME, OPMACRO, AVX512_OP, AVX2_OP)           \
    static inline __m512d NAME##_vec_f64_strN(__m512d a, __m512d b) {    \
        return _mm512_##AVX512_OP##_pd(a, b);                            \
    }

#elif defined(__AVX2__)

#define DEF_VEC_F64_OP_STRN(NAME, OPMACRO, AVX512_OP, AVX2_OP)           \
    static inline __m256d NAME##_vec_f64_strN(__m256d a, __m256d b) {    \
        return _mm256_##AVX2_OP##_pd(a, b);                              \
    }

#else

#define DEF_VEC_F64_OP_STRN(NAME, OPMACRO, AVX512_OP, AVX2_OP)           \
    static inline double NAME##_vec_f64_strN(double a, double b) {       \
        return a OPMACRO b;                                              \
    }

#endif


//---------------------------------------------------
// (6) スカラー演算用マクロ
//---------------------------------------------------
#define DEF_SCALAR_OP(NAME, OPMACRO)                              \
template <typename T>                                             \
static inline T NAME##_scalar(T a, T b) {                         \
    return a OPMACRO b;                                           \
}


//---------------------------------------------------
// (7) 最後に、上記マクロを使って add/sub/mul/div をまとめる
//---------------------------------------------------

//------------------------- ADD -------------------------
DEF_SCALAR_OP(add, OP_ADD)
DEF_VEC_F32_OP(add, OP_ADD, add, add, add)
DEF_VEC_F64_OP(add, OP_ADD, add, add, add)
DEF_VEC_F32_OP_STRN(add, OP_ADD, add, add)
DEF_VEC_F64_OP_STRN(add, OP_ADD, add, add)

//------------------------- SUB -------------------------
DEF_SCALAR_OP(sub, OP_SUB)
DEF_VEC_F32_OP(sub, OP_SUB, sub, sub, sub)
DEF_VEC_F64_OP(sub, OP_SUB, sub, sub, sub)
DEF_VEC_F32_OP_STRN(sub, OP_SUB, sub, sub)
DEF_VEC_F64_OP_STRN(sub, OP_SUB, sub, sub)

//------------------------- MUL -------------------------
DEF_SCALAR_OP(mul, OP_MUL)
DEF_VEC_F32_OP(mul, OP_MUL, mul, mul, mul)
DEF_VEC_F64_OP(mul, OP_MUL, mul, mul, mul)
DEF_VEC_F32_OP_STRN(mul, OP_MUL, mul, mul)
DEF_VEC_F64_OP_STRN(mul, OP_MUL, mul, mul)

//------------------------- DIV -------------------------
DEF_SCALAR_OP(div, OP_DIV)
DEF_VEC_F32_OP(div, OP_DIV, div, div, div)
DEF_VEC_F64_OP(div, OP_DIV, div, div, div)
DEF_VEC_F32_OP_STRN(div, OP_DIV, div, div)
DEF_VEC_F64_OP_STRN(div, OP_DIV, div, div)

#define IMPL_MAT_MAT_OP(OPNAME)                                 \
ZenuStatus zenu_compute_##OPNAME##_mat_mat_cpu(                 \
    void* dst,                                                  \
    const void* src1,                                           \
    const void* src2,                                           \
    int         stride_dst,                                     \
    int         stride_src1,                                    \
    int         stride_src2,                                    \
    size_t      n,                                              \
    ZenuDataType data_type                                      \
)                                                               \
{                                                               \
    if (!dst || !src1 || !src2 || n == 0) {                     \
        return ZenuStatus::InvalidArgument;                     \
    }                                                           \
                                                                \
    if (data_type == ZenuDataType::f32) {                       \
        iter_3buf(                                              \
            static_cast<const float*>(src1), stride_src1,       \
            static_cast<const float*>(src2), stride_src2,       \
            static_cast<float*>(dst), stride_dst,               \
            n,                                                  \
            OPNAME##_vec_f32,                                   \
            OPNAME##_vec_f32_strN,                              \
            OPNAME##_scalar<float>                              \
        );                                                      \
    }                                                           \
    else if (data_type == ZenuDataType::f64) {                  \
        iter_3buf(                                              \
            static_cast<const double*>(src1), stride_src1,      \
            static_cast<const double*>(src2), stride_src2,      \
            static_cast<double*>(dst), stride_dst,              \
            n,                                                  \
            OPNAME##_vec_f64,                                   \
            OPNAME##_vec_f64_strN,                              \
            OPNAME##_scalar<double>                             \
        );                                                      \
    }                                                           \
    else {                                                      \
        return ZenuStatus::InvalidArgument;                     \
    }                                                           \
                                                                \
    return ZenuStatus::Success;                                 \
}

extern "C" {
    IMPL_MAT_MAT_OP(add)
    IMPL_MAT_MAT_OP(sub)
    IMPL_MAT_MAT_OP(mul)
    IMPL_MAT_MAT_OP(div)
}
