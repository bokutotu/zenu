#include "zenu_compute.h"
#include <cstdlib>
#include <cstring>

ZenuStatus zenu_compute_malloc_cpu(void** ptr, int num_bytes)
{
    // コンパイル時マクロによるアライメント切り替え
#ifdef __AVX512F__
    constexpr std::size_t alignment = 64;  // AVX-512: 512-bit
#elif defined(__AVX2__)
    constexpr std::size_t alignment = 32;  // AVX2: 256-bit
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    constexpr std::size_t alignment = 16;  // NEON: 128-bit
#else
    // それ以外 (SSEやNon-SIMDなど) なら16や8を選ぶ等
    // 必要に応じて変更
    constexpr std::size_t alignment = 16;  
#endif

    // aligned_alloc は C11関数:
    //   void* aligned_alloc(std::size_t alignment, std::size_t size);
    // alignment は 2の冪であり、かつ size は alignment の倍数でなければならない。
    // 必要に応じて num_bytes を alignment 境界に切り上げる
    std::size_t adjusted_size = (num_bytes + alignment - 1) & ~(alignment - 1);

    // 実際の確保
    *ptr = aligned_alloc(alignment, adjusted_size);
    if (*ptr == nullptr) {
        return ZenuStatus::OutOfMemory;
    }

    return ZenuStatus::Success;
}

void zenu_compute_free_cpu(void* ptr) {
    free(ptr);
}

void zenu_compute_copy_cpu(void* dst, void* src, int num_bytes) {
    // TODO: Copy memory on the CPU
}

void zenu_compute_set_cpu(void* dst, int value, int num_bytes) {
    memset(dst, value, num_bytes);
}
