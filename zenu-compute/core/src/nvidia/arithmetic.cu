/**
 * @filearithmetic.cu
 * @brief "nvidia" GPU implementations of the arithmetic functions.
 *
 * メモリ転送は行わず、入出力ポインタ(dst, src1, src2等)は
 * すべてGPU上のデバイスメモリを指していると仮定。
 * （ホスト側メモリへのコピーや割り当ては一切行わない。）
 *
 */

#include "zenu_compute_type.h"
#include "utils.h"
#include "zenu_compute_arithmetic.h"
#include "iter_macro.h"
#include <cuda_runtime.h>
#include <stdio.h>

//---------------------------------------------
// Functors for +, -, *, /
//---------------------------------------------
struct AddOp {
    __device__ __host__ float  operator()(float a, float b)  const { return a + b; }
    __device__ __host__ double operator()(double a, double b)const { return a + b; }
};
struct SubOp {
    __device__ __host__ float  operator()(float a, float b)  const { return a - b; }
    __device__ __host__ double operator()(double a, double b)const { return a - b; }
};
struct MulOp {
    __device__ __host__ float  operator()(float a, float b)  const { return a * b; }
    __device__ __host__ double operator()(double a, double b)const { return a * b; }
};
struct DivOp {
    __device__ __host__ float  operator()(float a, float b)  const { return a / b; }
    __device__ __host__ double operator()(double a, double b)const { return a / b; }
};

//---------------------------------------------
// ADD
//---------------------------------------------
ZENU_NVIDIA_BINARY_OP(zenu_compute_add_mat_mat_nvidia,            AddOp)
ZENU_NVIDIA_SCALAR_OP(zenu_compute_add_mat_scalar_ptr_nvidia,     AddOp)
ZENU_NVIDIA_ASSIGN_OP(zenu_compute_add_mat_mat_assign_nvidia,     AddOp)
ZENU_NVIDIA_ASSIGN_SCALAR_OP(zenu_compute_add_mat_scalar_ptr_assign_nvidia, AddOp)

//---------------------------------------------
// SUB
//---------------------------------------------
ZENU_NVIDIA_BINARY_OP(zenu_compute_sub_mat_mat_nvidia,            SubOp)
ZENU_NVIDIA_SCALAR_OP(zenu_compute_sub_mat_scalar_ptr_nvidia,     SubOp)
ZENU_NVIDIA_ASSIGN_OP(zenu_compute_sub_mat_mat_assign_nvidia,     SubOp)
ZENU_NVIDIA_ASSIGN_SCALAR_OP(zenu_compute_sub_mat_scalar_ptr_assign_nvidia, SubOp)

//---------------------------------------------
// MUL
//---------------------------------------------
ZENU_NVIDIA_BINARY_OP(zenu_compute_mul_mat_mat_nvidia,            MulOp)
ZENU_NVIDIA_SCALAR_OP(zenu_compute_mul_mat_scalar_ptr_nvidia,     MulOp)
ZENU_NVIDIA_ASSIGN_OP(zenu_compute_mul_mat_mat_assign_nvidia,     MulOp)
ZENU_NVIDIA_ASSIGN_SCALAR_OP(zenu_compute_mul_mat_scalar_ptr_assign_nvidia, MulOp)

//---------------------------------------------
// DIV
//---------------------------------------------
ZENU_NVIDIA_BINARY_OP(zenu_compute_div_mat_mat_nvidia,            DivOp)
ZENU_NVIDIA_SCALAR_OP(zenu_compute_div_mat_scalar_ptr_nvidia,     DivOp)
ZENU_NVIDIA_ASSIGN_OP(zenu_compute_div_mat_mat_assign_nvidia,     DivOp)
ZENU_NVIDIA_ASSIGN_SCALAR_OP(zenu_compute_div_mat_scalar_ptr_assign_nvidia, DivOp)
