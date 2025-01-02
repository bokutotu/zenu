#include "zenu_compute.h"
#include "iter.h"

#if defined(__AVX512F__)
#  include <immintrin.h>
#  define SIMD_AVX512
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define SIMD_AVX2
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define SIMD_NEON
#else
#  define SIMD_NONE
#endif

template <typename T>
static inline T add_scalar(T a, T b) { return a + b; }

#if defined(SIMD_AVX512)
static inline __m512 add_vec_f32(__m512 a, __m512 b) { return _mm512_add_ps(a, b); }
#elif defined(SIMD_AVX2)
static inline __m256 add_vec_f32(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
#elif defined(SIMD_NEON)
static inline float32x4_t add_vec_f32(float32x4_t a, float32x4_t b) { return vaddq_f32(a, b); }
#else
static inline float add_vec_f32(float a, float b) { return a + b; }
#endif

#if defined(SIMD_AVX512)
static inline __m512d add_vec_f64(__m512d a, __m512d b) { return _mm512_add_pd(a, b); }
#elif defined(SIMD_AVX2)
static inline __m256d add_vec_f64(__m256d a, __m256d b) { return _mm256_add_pd(a, b); }
#elif defined(SIMD_NEON)
static inline float64x2_t add_vec_f64(float64x2_t a, float64x2_t b) { return vaddq_f64(a, b); }
#else
static inline double add_vec_f64(double a, double b) { return a + b; }
#endif

#if defined(SIMD_AVX512)
static inline __m512 add_vec_f32_strN(__m512 a, __m512 b) { return _mm512_add_ps(a, b); }
#elif defined(SIMD_AVX2)
static inline __m256 add_vec_f32_strN(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
#else
static inline float add_vec_f32_strN(float a, float b) { return a + b; }
#endif

#if defined(SIMD_AVX512)
static inline __m512d add_vec_f64_strN(__m512d a, __m512d b) { return _mm512_add_pd(a, b); }
#elif defined(SIMD_AVX2)
static inline __m256d add_vec_f64_strN(__m256d a, __m256d b) { return _mm256_add_pd(a, b); }
#else
static inline double add_vec_f64_strN(double a, double b) { return a + b; }
#endif

extern "C"
ZenuStatus zenu_compute_add_mat_mat_cpu(
    void*       dst,
    const void* src1,
    const void* src2,
    int         stride_dst,
    int         stride_src1,
    int         stride_src2,
    size_t      n,
    ZenuDataType data_type
)
{
    if (!dst || !src1 || !src2 || n == 0) {
        return ZenuStatus::InvalidArgument;
    }

    if (data_type == ZenuDataType::f32)
    {
        iter_3buf(
            static_cast<const float*>(src1), stride_src1,
            static_cast<const float*>(src2), stride_src2,
            static_cast<float*>(dst), stride_dst,
            n,
            add_vec_f32,
            add_vec_f32_strN,
            add_scalar<float>
        );
    }
    else if (data_type == ZenuDataType::f64)
    {
        iter_3buf(
            static_cast<const double*>(src1), stride_src1,
            static_cast<const double*>(src2), stride_src2,
            static_cast<double*>(dst), stride_dst,
            n,
            add_vec_f64,
            add_vec_f64_strN,
            add_scalar<double>
        );
    }
    else {
        return ZenuStatus::InvalidArgument;
    }

    return ZenuStatus::Success;
}
