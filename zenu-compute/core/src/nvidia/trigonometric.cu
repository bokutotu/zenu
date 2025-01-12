#include "zenu_compute.h"
#include "iter_macro.h"
#include <cuda_runtime.h>
#include <math.h>

struct SinFunctor {
    __device__ __host__ inline float operator()(float x) const  { return sinf(x); }
    __device__ __host__ inline double operator()(double x) const { return sin(x); }
};

struct CosFunctor {
    __device__ __host__ inline float operator()(float x) const  { return cosf(x); }
    __device__ __host__ inline double operator()(double x) const { return cos(x); }
};

struct TanFunctor {
    __device__ __host__ inline float operator()(float x) const  { return tanf(x); }
    __device__ __host__ inline double operator()(double x) const { return tan(x); }
};

struct SinhFunctor {
    __device__ __host__ inline float operator()(float x) const  { return sinhf(x); }
    __device__ __host__ inline double operator()(double x) const { return sinh(x); }
};

struct CoshFunctor {
    __device__ __host__ inline float operator()(float x) const  { return coshf(x); }
    __device__ __host__ inline double operator()(double x) const { return cosh(x); }
};

struct TanhFunctor {
    __device__ __host__ inline float operator()(float x) const  { return tanhf(x); }
    __device__ __host__ inline double operator()(double x) const { return tanh(x); }
};

ZENU_NVIDIA_UNARY_OP(zenu_compute_sin_mat_nvidia, SinFunctor)
ZENU_NVIDIA_UNARY_ASSIGN_OP(zenu_compute_sin_mat_assign_nvidia, SinFunctor)

ZENU_NVIDIA_UNARY_OP(zenu_compute_cos_mat_nvidia, CosFunctor)
ZENU_NVIDIA_UNARY_ASSIGN_OP(zenu_compute_cos_mat_assign_nvidia, CosFunctor)

ZENU_NVIDIA_UNARY_OP(zenu_compute_tan_mat_nvidia, TanFunctor)
ZENU_NVIDIA_UNARY_ASSIGN_OP(zenu_compute_tan_mat_assign_nvidia, TanFunctor)

ZENU_NVIDIA_UNARY_OP(zenu_compute_sinh_mat_nvidia, SinhFunctor)
ZENU_NVIDIA_UNARY_ASSIGN_OP(zenu_compute_sinh_mat_assign_nvidia, SinhFunctor)

ZENU_NVIDIA_UNARY_OP(zenu_compute_cosh_mat_nvidia, CoshFunctor)
ZENU_NVIDIA_UNARY_ASSIGN_OP(zenu_compute_cosh_mat_assign_nvidia, CoshFunctor)

ZENU_NVIDIA_UNARY_OP(zenu_compute_tanh_mat_nvidia, TanhFunctor)
ZENU_NVIDIA_UNARY_ASSIGN_OP(zenu_compute_tanh_mat_assign_nvidia, TanhFunctor)

