/**
 * @file extras.cu
 * @brief NVIDIA GPU (CUDA) 実装による exp, ln, abs, clip, pow の関数群
 *
 *  - out-of-place:   dst = f(src)  
 *  - in-place:       dst = f(dst)  
 *
 *  上記スタイルの関数を、CUDA カーネルで実行する実装例を示します。
 *  追加の最適化・エラーハンドリングは、プロジェクトに応じて検討してください。
 */

#include <math.h>
#include <cuda_runtime.h>
#include "zenu_compute_type.h"
#include "zenu_compute_extras.h"
#include "utils.h"
#include "iter_macro.h"
#include <stddef.h>

/**
 * @brief exp 関数用 Functor (CUDA device)
 */
struct ExpFunctor
{
    __host__ __device__
    ExpFunctor() {}
    
    template<typename T>
    __host__ __device__
    T operator()(T x) const {
        return (T)exp((double)x);
    }
};

/**
 * @brief ln (log) 関数用 Functor (CUDA device)
 */
struct LnFunctor
{
    __host__ __device__
    LnFunctor() {}
    
    template<typename T>
    __host__ __device__
    T operator()(T x) const {
        return (T)log((double)x);
    }
};

/**
 * @brief abs 関数用 Functor (CUDA device)
 */
struct AbsFunctor
{
    __host__ __device__
    AbsFunctor() {}
    
    template<typename T>
    __host__ __device__
    T operator()(T x) const {
        return (T)(x < 0 ? -x : x);
    }
};

/**
 * @brief clip 関数用 Functor (CUDA device)
 *
 *        - x < mn の場合は mn
 *        - x > mx の場合は mx
 *        - それ以外は x
 */
template<typename T>
struct ClipFunctor
{
    T mn;
    T mx;

    __host__ __device__
    ClipFunctor(T min_val, T max_val) : mn(min_val), mx(max_val) {}
    
    __host__ __device__
    T operator()(T x) const {
        return (x < mn) ? mn : ((x > mx) ? mx : x);
    }
};

/**
 * @brief pow 関数用 Functor (CUDA device)
 *
 *        operator() は (base, exponent) を受け取る
 */
struct PowFunctor
{
    __host__ __device__
    PowFunctor() {}
    
    template<typename T>
    __host__ __device__
    T operator()(T base, T exponent) const {
        return (T)pow((double)base, (double)exponent);
    }
};

ZENU_NVIDIA_UNARY_OP(zenu_compute_exp_mat_nvidia, ExpFunctor)
ZENU_NVIDIA_UNARY_ASSIGN_OP(zenu_compute_exp_mat_assign_nvidia, ExpFunctor)

ZENU_NVIDIA_UNARY_OP(zenu_compute_ln_mat_nvidia, LnFunctor)
ZENU_NVIDIA_UNARY_ASSIGN_OP(zenu_compute_ln_mat_assign_nvidia, LnFunctor)

ZENU_NVIDIA_UNARY_OP(zenu_compute_abs_mat_nvidia, AbsFunctor)
ZENU_NVIDIA_UNARY_ASSIGN_OP(zenu_compute_abs_mat_assign_nvidia, AbsFunctor)

/**
 * @brief clip kernel
 */
template<typename T>
__global__ void ClipKernel(
    T* dst, const T* src, int sd, int ss,
    size_t n, T mn, T mx
)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T x = src[i * ss];
        if (x < mn) x = mn;
        if (x > mx) x = mx;
        dst[i * sd] = x;
    }
}

/**
 * @brief clip kernel (in-place)
 */
template<typename T>
__global__ void ClipAssignKernel(
    T* dst, int sd, size_t n, T mn, T mx
)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        T x = dst[i * sd];
        if (x < mn) x = mn;
        if (x > mx) x = mx;
        dst[i * sd] = x;
    }
}

/**
 * @brief 配列 src の各要素に対して clip(min_val, max_val) を適用 (GPU: out-of-place)
 */
ZenuStatus zenu_compute_clip_mat_nvidia(
    void* dst, const void* src,
    int stride_dst, int stride_src,
    size_t n, ZenuDataType dt,
    double min_val, double max_val
)
{
    if (!dst || !src) return InvalidArgument;
    if (n == 0) return Success;

    dim3 block(256);
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));

    if (dt == f32) {
        ClipKernel<float><<<grid, block>>>(
            (float*)dst, (const float*)src,
            stride_dst, stride_src,
            n, (float)min_val, (float)max_val
        );
    } else if (dt == f64) {
        ClipKernel<double><<<grid, block>>>(
            (double*)dst, (const double*)src,
            stride_dst, stride_src,
            n, min_val, max_val
        );
    } else {
        return InvalidArgument;
    }

    cudaError_t e = cudaDeviceSynchronize();
    return convertCudaError(e);
}

/**
 * @brief 配列 dst の各要素に対して clip(min_val, max_val) を適用 (GPU: in-place)
 */
ZenuStatus zenu_compute_clip_mat_assign_nvidia(
    void* dst,
    int stride_dst,
    size_t n, ZenuDataType dt,
    double min_val, double max_val
)
{
    if (!dst) return InvalidArgument;
    if (n == 0) return Success;

    dim3 block(256);
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));

    if (dt == f32) {
        ClipAssignKernel<float><<<grid, block>>>(
            (float*)dst, stride_dst,
            n, (float)min_val, (float)max_val
        );
    } else if (dt == f64) {
        ClipAssignKernel<double><<<grid, block>>>(
            (double*)dst, stride_dst,
            n, min_val, max_val
        );
    } else {
        return InvalidArgument;
    }

    cudaError_t e = cudaDeviceSynchronize();
    return convertCudaError(e);
}

ZENU_NVIDIA_SCALAR_OP(zenu_compute_pow_mat_nvidia, PowFunctor)
ZENU_NVIDIA_ASSIGN_SCALAR_OP(zenu_compute_pow_mat_assign_nvidia, PowFunctor)

