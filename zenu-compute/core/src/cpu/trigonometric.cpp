#include <math.h>
#include "zenu_compute_trigonometric.h"
#include "zenu_compute.h"
#include "utils.h"
#include "iter_macro.h"

ZENU_CPU_UNARY_OP(
    zenu_compute_sin_mat_cpu,     // 関数名 (out-of-place)
    sinf, sin                     // シングル精度/倍精度それぞれの関数
)

ZENU_CPU_UNARY_ASSIGN_OP(
    zenu_compute_sin_mat_assign_cpu,  // 関数名 (in-place)
    sinf, sin
)

ZENU_CPU_UNARY_OP(
    zenu_compute_cos_mat_cpu,
    cosf, cos
)

ZENU_CPU_UNARY_ASSIGN_OP(
    zenu_compute_cos_mat_assign_cpu,
    cosf, cos
)

ZENU_CPU_UNARY_OP(
    zenu_compute_tan_mat_cpu,
    tanf, tan
)

ZENU_CPU_UNARY_ASSIGN_OP(
    zenu_compute_tan_mat_assign_cpu,
    tanf, tan
)

ZENU_CPU_UNARY_OP(
    zenu_compute_sinh_mat_cpu,
    sinhf, sinh
)

ZENU_CPU_UNARY_ASSIGN_OP(
    zenu_compute_sinh_mat_assign_cpu,
    sinhf, sinh
)

ZENU_CPU_UNARY_OP(
    zenu_compute_cosh_mat_cpu,
    coshf, cosh
)

ZENU_CPU_UNARY_ASSIGN_OP(
    zenu_compute_cosh_mat_assign_cpu,
    coshf, cosh
)

ZENU_CPU_UNARY_OP(
    zenu_compute_tanh_mat_cpu,
    tanhf, tanh
)

ZENU_CPU_UNARY_ASSIGN_OP(
    zenu_compute_tanh_mat_assign_cpu,
    tanhf, tanh
)
