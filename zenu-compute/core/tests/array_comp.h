#pragma once

#include <cmath>
#include <cstddef>

/**
 * @brief 2 つの配列が等しいかどうかを比較する関数
 *
 * @param[in] a   比較対象の配列 a
 * @param[in] b   比較対象の配列 b
 * @param[in] n   配列の要素数
 * @param[in] tol 許容誤差
 * @return bool   2 つの配列が等しい場合は true、そうでない場合は false
 */
template <typename T>
inline bool array_compare(const T* a, const T* b, size_t n, T tol = 1e-5)
{
    for (size_t i = 0; i < n; ++i) {
        T diff = std::fabs(a[i] - b[i]);
        if (diff > tol) {
            return false;
        }
    }
    return true;
}
