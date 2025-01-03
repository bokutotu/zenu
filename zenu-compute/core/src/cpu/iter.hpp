#include <cstddef>
template <typename T, class Op>
void iter_1(T* data, size_t size, Op op) {
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        op(data[i]);
    }
}

template <typename T, class Op>
void iter_2(T* dst, const T* src, size_t size, Op op) {
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        dst[i] = op(src[i]);
    }
}

template <typename T, class Op>
void iter_2_assign(T* dst, const T* src, size_t size, Op op) {
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        dst[i] = op(dst[i], src[i]);
    }
}

template <typename T, class Op>
void iter_3(T* dst, const T* src1, const T* src2, size_t size, Op op) {
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        dst[i] = op(src1[i], src2[i]);
    }
}
