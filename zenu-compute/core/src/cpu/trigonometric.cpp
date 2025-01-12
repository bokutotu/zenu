#include <math.h>
#include "iter_macro.h"
#include <omp.h>

/*-----------------------------------------
 * 1) ファンクタ (三角, 双曲線) 定義
 *-----------------------------------------*/
struct SinFunctor {
    inline float operator()(float x) const  { return sinf(x); }
    inline double operator()(double x) const { return sin(x); }
};

struct CosFunctor {
    inline float operator()(float x) const  { return cosf(x); }
    inline double operator()(double x) const { return cos(x); }
};

struct TanFunctor {
    inline float operator()(float x) const  { return tanf(x); }
    inline double operator()(double x) const { return tan(x); }
};

struct SinhFunctor {
    inline float operator()(float x) const  { return sinhf(x); }
    inline double operator()(double x) const { return sinh(x); }
};

struct CoshFunctor {
    inline float operator()(float x) const  { return coshf(x); }
    inline double operator()(double x) const { return cosh(x); }
};

struct TanhFunctor {
    inline float operator()(float x) const  { return tanhf(x); }
    inline double operator()(double x) const { return tanh(x); }
};

/*-----------------------------------------
 * 2) マクロを用いて実装
 *    out-of-place: ZENU_CPU_UNARY_OP
 *    in-place:     ZENU_CPU_UNARY_ASSIGN_OP
 *-----------------------------------------*/

ZENU_CPU_UNARY_OP(zenu_compute_sin_mat_cpu, SinFunctor)
ZENU_CPU_UNARY_ASSIGN_OP(zenu_compute_sin_mat_assign_cpu, SinFunctor)

ZENU_CPU_UNARY_OP(zenu_compute_cos_mat_cpu, CosFunctor)
ZENU_CPU_UNARY_ASSIGN_OP(zenu_compute_cos_mat_assign_cpu, CosFunctor)

ZENU_CPU_UNARY_OP(zenu_compute_tan_mat_cpu, TanFunctor)
ZENU_CPU_UNARY_ASSIGN_OP(zenu_compute_tan_mat_assign_cpu, TanFunctor)

ZENU_CPU_UNARY_OP(zenu_compute_sinh_mat_cpu, SinhFunctor)
ZENU_CPU_UNARY_ASSIGN_OP(zenu_compute_sinh_mat_assign_cpu, SinhFunctor)

ZENU_CPU_UNARY_OP(zenu_compute_cosh_mat_cpu, CoshFunctor)
ZENU_CPU_UNARY_ASSIGN_OP(zenu_compute_cosh_mat_assign_cpu, CoshFunctor)

ZENU_CPU_UNARY_OP(zenu_compute_tanh_mat_cpu, TanhFunctor)
ZENU_CPU_UNARY_ASSIGN_OP(zenu_compute_tanh_mat_assign_cpu, TanhFunctor)
