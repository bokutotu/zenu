#include <math.h>                     // sinf, sin, cosf, cos, tanf, tan, etc.
#include "zenu_compute_trigonometric.h"
#include "zenu_compute.h"
#include "utils.h"                    // check_common_args()
#include "iter_macro.h"               // 先ほど追加したマクロを含む

/****************************************************************************
 *  Implementation of trigonometric/hyperbolic functions for CPU
 ****************************************************************************/

/*--------------------------------------
 * SIN
 *-------------------------------------*/
ZENU_CPU_UNARY_MATH_FUNC(
    zenu_compute_sin_mat_cpu,     // 関数名 (out-of-place)
    sinf, sin                     // シングル精度/倍精度それぞれの関数
)

ZENU_CPU_UNARY_MATH_FUNC_ASSIGN(
    zenu_compute_sin_mat_assign_cpu,  // 関数名 (in-place)
    sinf, sin
)

/*--------------------------------------
 * COS
 *-------------------------------------*/
ZENU_CPU_UNARY_MATH_FUNC(
    zenu_compute_cos_mat_cpu,
    cosf, cos
)

ZENU_CPU_UNARY_MATH_FUNC_ASSIGN(
    zenu_compute_cos_mat_assign_cpu,
    cosf, cos
)

/*--------------------------------------
 * TAN
 *-------------------------------------*/
ZENU_CPU_UNARY_MATH_FUNC(
    zenu_compute_tan_mat_cpu,
    tanf, tan
)

ZENU_CPU_UNARY_MATH_FUNC_ASSIGN(
    zenu_compute_tan_mat_assign_cpu,
    tanf, tan
)

/*--------------------------------------
 * SINH
 *-------------------------------------*/
ZENU_CPU_UNARY_MATH_FUNC(
    zenu_compute_sinh_mat_cpu,
    sinhf, sinh
)

ZENU_CPU_UNARY_MATH_FUNC_ASSIGN(
    zenu_compute_sinh_mat_assign_cpu,
    sinhf, sinh
)

/*--------------------------------------
 * COSH
 *-------------------------------------*/
ZENU_CPU_UNARY_MATH_FUNC(
    zenu_compute_cosh_mat_cpu,
    coshf, cosh
)

ZENU_CPU_UNARY_MATH_FUNC_ASSIGN(
    zenu_compute_cosh_mat_assign_cpu,
    coshf, cosh
)

/*--------------------------------------
 * TANH
 *-------------------------------------*/
ZENU_CPU_UNARY_MATH_FUNC(
    zenu_compute_tanh_mat_cpu,
    tanhf, tanh
)

ZENU_CPU_UNARY_MATH_FUNC_ASSIGN(
    zenu_compute_tanh_mat_assign_cpu,
    tanhf, tanh
)


