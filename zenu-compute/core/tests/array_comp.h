#pragma once

#include <cmath>
#include <cstddef>

/**
 * @brief  単精度浮動小数点配列を比較する
 *
 * @param[in] a   配列 a (要素数 n)
 * @param[in] b   配列 b (要素数 n)
 * @param[in] n   配列の要素数
 * @param[in] tol 許容誤差
 * @return true   すべての要素が tol 以内で近似一致している
 * @return false  一部の要素が tol を超えている
 */
inline bool zenu_compare_array_f32(const float* a, const float* b, size_t n, float tol = 1e-5f)
{
    for (size_t i = 0; i < n; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > tol) {
            return false;
        }
    }
    return true;
}

/**
 * @brief  倍精度浮動小数点配列を比較する
 *
 * @param[in] a   配列 a (要素数 n)
 * @param[in] b   配列 b (要素数 n)
 * @param[in] n   配列の要素数
 * @param[in] tol 許容誤差
 * @return true   すべての要素が tol 以内で近似一致している
 * @return false  一部の要素が tol を超えている
 */
inline bool zenu_compare_array_f64(const double* a, const double* b, size_t n, double tol = 1e-9)
{
    for (size_t i = 0; i < n; ++i) {
        double diff = std::fabs(a[i] - b[i]);
        if (diff > tol) {
            return false;
        }
    }
    return true;
}

