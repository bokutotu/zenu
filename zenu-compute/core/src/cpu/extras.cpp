/**
 * @file extras.cpp
 * @brief CPU (OpenMP) 実装による exp, ln, abs, clip, pow の関数群
 *
 *  - out-of-place:   dst = f(src)  
 *  - in-place:       dst = f(dst)  
 *
 *  上記スタイルの関数を、OpenMP を用いて並列化した実装を示します。
 *  追加の最適化オプションなどは、プロジェクトに応じて検討・調整してください。
 */

#include <math.h>
#include "zenu_compute_type.h"
#include "iter_macro.h"
#include <stddef.h>

/**
 * @brief exp 関数用 Functor
 */
struct ExpFunctor {
    template<typename T>
    T operator()(T x) const { return (T)exp((double)x); }
};

/**
 * @brief ln (log) 関数用 Functor
 */
struct LnFunctor {
    template<typename T>
    T operator()(T x) const { return (T)log((double)x); }
};

/**
 * @brief abs 関数用 Functor
 */
struct AbsFunctor {
    template<typename T>
    T operator()(T x) const { return (T)fabs((double)x); }
};

/**
 * @brief clip 関数用 Functor
 *
 *        - x < mn の場合は mn
 *        - x > mx の場合は mx
 *        - それ以外は x
 */
template<typename T>
struct ClipFunctor {
    T mn;
    T mx;
    ClipFunctor(T min_val, T max_val) : mn(min_val), mx(max_val) {}
    T operator()(T x) const {
        return (x < mn) ? mn : ((x > mx) ? mx : x);
    }
};

/**
 * @brief pow 関数用 Functor
 *
 *        operator() は (base, exponent) を受け取る
 */
struct PowFunctor {
    template<typename T>
    T operator()(T base, T exponent) const {
        return (T)pow((double)base, (double)exponent);
    }
};

ZENU_CPU_UNARY_OP(zenu_compute_exp_mat_cpu, ExpFunctor)
ZENU_CPU_UNARY_ASSIGN_OP(zenu_compute_exp_mat_assign_cpu, ExpFunctor)

ZENU_CPU_UNARY_OP(zenu_compute_ln_mat_cpu, LnFunctor)
ZENU_CPU_UNARY_ASSIGN_OP(zenu_compute_ln_mat_assign_cpu, LnFunctor)

ZENU_CPU_UNARY_OP(zenu_compute_abs_mat_cpu, AbsFunctor)
ZENU_CPU_UNARY_ASSIGN_OP(zenu_compute_abs_mat_assign_cpu, AbsFunctor)

ZenuStatus zenu_compute_clip_mat_cpu(
    void* dst, const void* src,
    int stride_dst, int stride_src,
    size_t n, ZenuDataType dt,
    double min_val, double max_val
)
{
    if (!dst || !src) return InvalidArgument;
    if (n == 0) return Success;

    if (dt == f32) {
        float*       pDst = (float*)dst;
        const float* pSrc = (const float*)src;
        ClipFunctor<float> functor((float)min_val, (float)max_val);

#pragma omp parallel for simd
        for (size_t i = 0; i < n; i++) {
            pDst[i * stride_dst] = functor(pSrc[i * stride_src]);
        }
    } else if (dt == f64) {
        double*       pDst = (double*)dst;
        const double* pSrc = (const double*)src;
        ClipFunctor<double> functor(min_val, max_val);

#pragma omp parallel for simd
        for (size_t i = 0; i < n; i++) {
            pDst[i * stride_dst] = functor(pSrc[i * stride_src]);
        }
    } else {
        return InvalidArgument;
    }
    return Success;
}

ZenuStatus zenu_compute_clip_mat_assign_cpu(
    void* dst,
    int stride_dst,
    size_t n, ZenuDataType dt,
    double min_val, double max_val
)
{
    if (!dst) return InvalidArgument;
    if (n == 0) return Success;

    if (dt == f32) {
        float* pDst = (float*)dst;
        ClipFunctor<float> functor((float)min_val, (float)max_val);

#pragma omp parallel for simd
        for (size_t i = 0; i < n; i++) {
            pDst[i * stride_dst] = functor(pDst[i * stride_dst]);
        }
    } else if (dt == f64) {
        double* pDst = (double*)dst;
        ClipFunctor<double> functor(min_val, max_val);

#pragma omp parallel for simd
        for (size_t i = 0; i < n; i++) {
            pDst[i * stride_dst] = functor(pDst[i * stride_dst]);
        }
    } else {
        return InvalidArgument;
    }
    return Success;
}

ZENU_CPU_SCALAR_OP(zenu_compute_pow_mat_cpu, PowFunctor)
ZENU_CPU_ASSIGN_SCALAR_OP(zenu_compute_pow_mat_assign_cpu, PowFunctor)
