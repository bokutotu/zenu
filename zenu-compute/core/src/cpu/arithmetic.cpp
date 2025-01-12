/**
 * @file arithmetic.cpp
 * @brief Implementation of arithmetic functions (add/sub/mul/div) on CPU
 */

#include "zenu_compute.h"
#include "utils.h"
#include "iter_macro.h"

#include <omp.h>                     // OpenMP
#include <string.h>
#include <stdio.h>

//========== 1) 演算ファンクタ定義 ==========
struct AddFunctor {
    template<typename T> inline T operator()(T x, T y) const { return x + y; }
};
struct SubFunctor {
    template<typename T> inline T operator()(T x, T y) const { return x - y; }
};
struct MulFunctor {
    template<typename T> inline T operator()(T x, T y) const { return x * y; }
};
struct DivFunctor {
    template<typename T> inline T operator()(T x, T y) const { return x / y; }
};

//========== 2) ADD (加算) の CPU 実装 ==========
ZENU_CPU_BINARY_OP(zenu_compute_add_mat_mat_cpu, AddFunctor)
ZENU_CPU_SCALAR_OP(zenu_compute_add_mat_scalar_ptr_cpu, AddFunctor)
ZENU_CPU_ASSIGN_OP(zenu_compute_add_mat_mat_assign_cpu, AddFunctor)
ZENU_CPU_ASSIGN_SCALAR_OP(zenu_compute_add_mat_scalar_ptr_assign_cpu, AddFunctor)

//========== 3) SUB (減算) の CPU 実装 ==========
ZENU_CPU_BINARY_OP(zenu_compute_sub_mat_mat_cpu, SubFunctor)
ZENU_CPU_SCALAR_OP(zenu_compute_sub_mat_scalar_ptr_cpu, SubFunctor)
ZENU_CPU_ASSIGN_OP(zenu_compute_sub_mat_mat_assign_cpu, SubFunctor)
ZENU_CPU_ASSIGN_SCALAR_OP(zenu_compute_sub_mat_scalar_ptr_assign_cpu, SubFunctor)

//========== 4) MUL (乗算) の CPU 実装 ==========
ZENU_CPU_BINARY_OP(zenu_compute_mul_mat_mat_cpu, MulFunctor)
ZENU_CPU_SCALAR_OP(zenu_compute_mul_mat_scalar_ptr_cpu, MulFunctor)
ZENU_CPU_ASSIGN_OP(zenu_compute_mul_mat_mat_assign_cpu, MulFunctor)
ZENU_CPU_ASSIGN_SCALAR_OP(zenu_compute_mul_mat_scalar_ptr_assign_cpu, MulFunctor)

//========== 5) DIV (除算) の CPU 実装 ==========
ZENU_CPU_BINARY_OP(zenu_compute_div_mat_mat_cpu, DivFunctor)
ZENU_CPU_SCALAR_OP(zenu_compute_div_mat_scalar_ptr_cpu, DivFunctor)
ZENU_CPU_ASSIGN_OP(zenu_compute_div_mat_mat_assign_cpu, DivFunctor)
ZENU_CPU_ASSIGN_SCALAR_OP(zenu_compute_div_mat_scalar_ptr_assign_cpu, DivFunctor)
