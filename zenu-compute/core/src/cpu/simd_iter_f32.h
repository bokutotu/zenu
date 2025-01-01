#pragma once

#include <cstddef>
#include <cstdint>
#include <omp.h>

/**
 * @brief is_aligned
 *        アドレスが指定した align バイト境界に乗っているか判定する
 *
 * @param ptr   チェック対象ポインタ
 * @param align アラインメント (例: 64)
 * @return true  if ptr が align バイト境界
 *         false if ptr がずれている
 */
static inline bool is_aligned(const void* ptr, std::size_t align)
{
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    return (addr % align) == 0;
}

//============================================================
// SIMD 判定
//============================================================
#if defined(__AVX512F__)
#  include <immintrin.h>
#  define SIMD_AVX512
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define SIMD_AVX2
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define SIMD_NEON
#else
#  define SIMD_NONE
#endif

//============================================================
// AVX2/AVX512用 gather ユーティリティ (stride≠1)
//============================================================
#if defined(SIMD_AVX512) || defined(SIMD_AVX2)
/**
 * @brief gather_f32_avx2: AVX2(256-bit)用 gather
 *
 * @param base     ベースポインタ(float配列)
 * @param indices  インデックス配列(要素数8)
 * @param count    デフォルト8
 * @return __m256  8要素まとめたベクトル
 */
static inline __m256 gather_f32_avx2(const float* base, const int* indices, int count=8)
{
    __m256i vi = _mm256_loadu_si256((const __m256i*)indices);
    return _mm256_i32gather_ps(base, vi, 4); // 4=sizeof(float
}

#if defined(SIMD_AVX512)
/**
 * @brief gather_f32_avx512: AVX512(512-bit)用 gather
 *
 * @param base     ベースポインタ(float配列)
 * @param indices  インデックス配列(要素数16)
 * @param count    デフォルト16
 * @return __m512  16要素まとめたベクトル
 */
static inline __m512 gather_f32_avx512(const float* base, const int* indices, int count=16)
{
    __m512i vi = _mm512_loadu_si512((const void*)indices);
    return _mm512_i32gather_ps(vi, base, 4);
}
#endif
#endif // SIMD_AVX2/AVX512

//============================================================
// (1) 1バッファ, stride=1
//============================================================
/**
 * @brief [1バッファ, stride=1]
 *        buf[i] = opScalar(buf[i]) / or opVec(buf[i])
 *        aligned/unaligned を先頭アドレスで判定
 *
 * @tparam OpVecOne    __m256 operator()(__m256) など
 * @tparam OpScalarOne float operator()(float)
 *
 * @param buf [in/out]   要素数 n の float配列
 * @param n   [in]       要素数
 * @param opVec    SIMD演算
 * @param opScalar スカラー演算
 */
template<class OpVecOne, class OpScalarOne>
void iter_1buf_f32_str1_vec_omp(
    float* buf,
    std::size_t n,
    OpVecOne   opVec,
    OpScalarOne opScalar)
{
#if defined(SIMD_AVX512)
    constexpr std::size_t alignment = 64; 
    constexpr std::size_t step = 16; 
#elif defined(SIMD_AVX2)
    constexpr std::size_t alignment = 32;
    constexpr std::size_t step = 8;
#elif defined(SIMD_NEON)
    constexpr std::size_t alignment = 16;
    constexpr std::size_t step = 4;
#else
    constexpr std::size_t alignment = 16;
    constexpr std::size_t step = 1;
#endif

    bool aligned_head = is_aligned(buf, alignment);

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (std::size_t blockStart = 0; blockStart < n; blockStart += step)
        {
            std::size_t remain = n - blockStart;
            std::size_t chunk  = (remain >= step)? step : remain;
            if (chunk < step) {
                // 端数スカラー
                for (std::size_t i=0; i<chunk; i++) {
                    buf[blockStart + i] = opScalar(buf[blockStart + i]);
                }
                continue;
            }

#if defined(SIMD_AVX512)
            if (aligned_head) {
                __m512 x = _mm512_load_ps(&buf[blockStart]);
                x = opVec(x);
                _mm512_store_ps(&buf[blockStart], x);
            } else {
                __m512 x = _mm512_loadu_ps(&buf[blockStart]);
                x = opVec(x);
                _mm512_storeu_ps(&buf[blockStart], x);
            }

#elif defined(SIMD_AVX2)
            if (aligned_head) {
                __m256 x = _mm256_load_ps(&buf[blockStart]);
                x = opVec(x);
                _mm256_store_ps(&buf[blockStart], x);
            } else {
                __m256 x = _mm256_loadu_ps(&buf[blockStart]);
                x = opVec(x);
                _mm256_storeu_ps(&buf[blockStart], x);
            }

#elif defined(SIMD_NEON)
            float32x4_t x = vld1q_f32(&buf[blockStart]);
            x = opVec(x);
            vst1q_f32(&buf[blockStart], x);

#else
            // fallback
            for(std::size_t i=0; i<chunk; i++){
                buf[blockStart + i] = opScalar(buf[blockStart + i]);
            }
#endif
        }
    }
}

//============================================================
// (2) 1バッファ, stride≠1
//============================================================
/**
 * @brief [1バッファ, stride!=1]
 *        buf[i*stride] = opScalar(buf[i*stride]) or gather/scatter
 */
template<class OpVecOne, class OpScalarOne>
void iter_1buf_f32_strN_vec_omp(
    float* buf,
    std::size_t n,
    std::size_t stride,
    OpVecOne   opVec,
    OpScalarOne opScalar)
{
#if defined(SIMD_AVX512)
    const std::size_t step = 16;
#elif defined(SIMD_AVX2)
    const std::size_t step = 8;
#else
    const std::size_t step = 1; 
#endif

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (std::size_t blockStart = 0; blockStart < n; blockStart += step)
        {
            std::size_t remain = n - blockStart;
            if (remain >= step && step>1) {
#if defined(SIMD_AVX512)
                // gather
                int idx[16];
                for(int i=0; i<16; i++){
                    idx[i] = int((blockStart + i)*stride);
                }
                __m512 x = gather_f32_avx512(buf, idx, 16);
                x = opVec(x);
                // scatter
                __m512i vi = _mm512_loadu_si512((const void*)idx);
                _mm512_i32scatter_ps(buf, vi, x, 4);

#elif defined(SIMD_AVX2)
                int idx[8];
                for(int i=0; i<8; i++){
                    idx[i] = int((blockStart + i)*stride);
                }
                __m256 x = gather_f32_avx2(buf, idx, 8);
                x = opVec(x);
                // scatter
                float tmp[8];
                _mm256_storeu_ps(tmp, x);
                for(int i=0; i<8; i++){
                    buf[idx[i]] = tmp[i];
                }
#endif
            } else {
                // fallback
                std::size_t actual = (remain<step)? remain : step;
                for(std::size_t i=0; i<actual; i++){
                    std::size_t pos = (blockStart + i)*stride;
                    buf[pos] = opScalar(buf[pos]);
                }
            }
        }
    }
}

//============================================================
// (3) 2バッファ: stride=1
//============================================================
/**
 * @brief [2バッファ(inA, outB), stride=1]
 *        outB[i] = opScalar(inA[i], outB[i]) など
 */
template<class OpVec, class OpScalar>
void iter_2buf_f32_str1_vec_omp(
    const float* inA,
    float*       outB,
    std::size_t  n,
    OpVec        opVec,     ///< ex: __m256 operator()(__m256 a, __m256 b)
    OpScalar     opScalar   ///< ex: float operator()(float a, float b)
)
{
#if defined(SIMD_AVX512)
    constexpr std::size_t alignment = 64;
    constexpr std::size_t step = 16;
#elif defined(SIMD_AVX2)
    constexpr std::size_t alignment = 32;
    constexpr std::size_t step = 8;
#elif defined(SIMD_NEON)
    constexpr std::size_t alignment = 16;
    constexpr std::size_t step = 4;
#else
    constexpr std::size_t alignment = 16;
    constexpr std::size_t step = 1;
#endif

    bool alignedA = is_aligned(inA,  alignment);
    bool alignedB = is_aligned(outB, alignment);
    bool do_aligned = (alignedA && alignedB);

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (std::size_t blockStart = 0; blockStart < n; blockStart += step)
        {
            std::size_t remain = n - blockStart;
            std::size_t chunk  = (remain >= step)? step : remain;

            if (chunk < step) {
                // 端数 スカラー fallback
                for(std::size_t i=0; i<chunk; i++){
                    float a = inA[blockStart + i];
                    float b = outB[blockStart + i];
                    // outB[i] = opScalar(a,b)
                    outB[blockStart + i] = opScalar(a, b);
                }
                continue;
            }

#if defined(SIMD_AVX512)
            if (do_aligned) {
                // aligned load
                __m512 va = _mm512_load_ps(&inA[blockStart]);
                __m512 vb = _mm512_load_ps(&outB[blockStart]);
                __m512 vb_new = opVec(va, vb);
                _mm512_store_ps(&outB[blockStart], vb_new);
            } else {
                __m512 va = _mm512_loadu_ps(&inA[blockStart]);
                __m512 vb = _mm512_loadu_ps(&outB[blockStart]);
                __m512 vb_new = opVec(va, vb);
                _mm512_storeu_ps(&outB[blockStart], vb_new);
            }

#elif defined(SIMD_AVX2)
            if (do_aligned) {
                __m256 va = _mm256_load_ps(&inA[blockStart]);
                __m256 vb = _mm256_load_ps(&outB[blockStart]);
                __m256 vb_new = opVec(va, vb);
                _mm256_store_ps(&outB[blockStart], vb_new);
            } else {
                __m256 va = _mm256_loadu_ps(&inA[blockStart]);
                __m256 vb = _mm256_loadu_ps(&outB[blockStart]);
                __m256 vb_new = opVec(va, vb);
                _mm256_storeu_ps(&outB[blockStart], vb_new);
            }

#elif defined(SIMD_NEON)
            float32x4_t va = vld1q_f32(&inA[blockStart]);
            float32x4_t vb = vld1q_f32(&outB[blockStart]);
            float32x4_t vb_new = opVec(va, vb);
            vst1q_f32(&outB[blockStart], vb_new);

#else
            // fallback
            for(std::size_t i=0; i<chunk; i++){
                float a = inA[blockStart + i];
                float b = outB[blockStart + i];
                outB[blockStart + i] = opScalar(a, b);
            }
#endif
        }
    }
}

//============================================================
// (4) 2バッファ: stride≠1
//============================================================
/**
 * @brief [2バッファ(inA, outB), stride!=1]
 *        outB[idxB] = opScalar(inA[idxA], outB[idxB]) など
 */
template<class OpVec, class OpScalar>
void iter_2buf_f32_strN_vec_omp(
    const float* inA,  std::size_t strideA,
    float*       outB, std::size_t strideB,
    std::size_t  n,
    OpVec        opVec,
    OpScalar     opScalar
)
{
#if defined(SIMD_AVX512)
    const std::size_t step = 16;
#elif defined(SIMD_AVX2)
    const std::size_t step = 8;
#else
    const std::size_t step = 1;
#endif

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (std::size_t blockStart = 0; blockStart < n; blockStart += step)
        {
            std::size_t remain = n - blockStart;
            std::size_t chunk  = (remain >= step)? step : remain;

            if (chunk < step || step==1) {
                // fallbackスカラー
                for(std::size_t i=0; i<chunk; i++){
                    std::size_t idxA = (blockStart + i)*strideA;
                    std::size_t idxB = (blockStart + i)*strideB;
                    float a = inA[idxA];
                    float b = outB[idxB];
                    outB[idxB] = opScalar(a, b);
                }
                continue;
            }

#if defined(SIMD_AVX512)
            int idxA_[16], idxB_[16];
            for(int i=0; i<16; i++){
                idxA_[i] = int((blockStart + i)*strideA);
                idxB_[i] = int((blockStart + i)*strideB);
            }
            // gather
            __m512 va = gather_f32_avx512(inA,  idxA_, 16);
            __m512 vb = gather_f32_avx512(outB, idxB_, 16);
            // 演算
            __m512 vb_new = opVec(va, vb);
            // scatter
            __m512i viB = _mm512_loadu_si512((const void*)idxB_);
            _mm512_i32scatter_ps(outB, viB, vb_new, 4);

#elif defined(SIMD_AVX2)
            int idxA_[8], idxB_[8];
            for(int i=0; i<8; i++){
                idxA_[i] = int((blockStart + i)*strideA);
                idxB_[i] = int((blockStart + i)*strideB);
            }
            __m256 va = gather_f32_avx2(inA, idxA_, 8);
            __m256 vb = gather_f32_avx2(outB, idxB_, 8);
            __m256 vb_new = opVec(va, vb);
            // scatter
            float tmp[8];
            _mm256_storeu_ps(tmp, vb_new);
            for(int i=0; i<8; i++){
                outB[idxB_[i]] = tmp[i];
            }
#endif
        }
    }
}

//============================================================
// (5) 3バッファ: stride=1
//============================================================
/**
 * @brief [3バッファ(inA, inB) -> outC, stride=1]
 *        outC[i] = opScalar(inA[i], inB[i]) / or opVec(va,vb)
 */
template<class OpVec, class OpScalar>
void iter_3buf_f32_str1_vec_omp(
    const float* inA,
    const float* inB,
    float*       outC,
    std::size_t  n,
    OpVec        opVec,     ///< ex: __m256 operator()(__m256, __m256)
    OpScalar     opScalar   ///< ex: float operator()(float, float)
)
{
#if defined(SIMD_AVX512)
    constexpr std::size_t alignment = 64;
    constexpr std::size_t step = 16;
#elif defined(SIMD_AVX2)
    constexpr std::size_t alignment = 32;
    constexpr std::size_t step = 8;
#elif defined(SIMD_NEON)
    constexpr std::size_t alignment = 16;
    constexpr std::size_t step = 4;
#else
    constexpr std::size_t alignment = 16;
    constexpr std::size_t step = 1;
#endif

    bool alignedA = is_aligned(inA,  alignment);
    bool alignedB = is_aligned(inB,  alignment);
    bool alignedC = is_aligned(outC, alignment);
    bool do_aligned = (alignedA && alignedB && alignedC);

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for(std::size_t blockStart=0; blockStart < n; blockStart += step)
        {
            std::size_t remain = n - blockStart;
            std::size_t chunk  = (remain >= step)? step : remain;
            if (chunk < step) {
                // fallback
                for(std::size_t i=0; i<chunk; i++){
                    float a = inA[blockStart + i];
                    float b = inB[blockStart + i];
                    outC[blockStart + i] = opScalar(a, b);
                }
                continue;
            }

#if defined(SIMD_AVX512)
            if (do_aligned) {
                __m512 va = _mm512_load_ps(&inA[blockStart]);
                __m512 vb = _mm512_load_ps(&inB[blockStart]);
                __m512 vc = opVec(va, vb);
                _mm512_store_ps(&outC[blockStart], vc);
            } else {
                __m512 va = _mm512_loadu_ps(&inA[blockStart]);
                __m512 vb = _mm512_loadu_ps(&inB[blockStart]);
                __m512 vc = opVec(va, vb);
                _mm512_storeu_ps(&outC[blockStart], vc);
            }

#elif defined(SIMD_AVX2)
            if (do_aligned) {
                __m256 va = _mm256_load_ps(&inA[blockStart]);
                __m256 vb = _mm256_load_ps(&inB[blockStart]);
                __m256 vc = opVec(va, vb);
                _mm256_store_ps(&outC[blockStart], vc);
            } else {
                __m256 va = _mm256_loadu_ps(&inA[blockStart]);
                __m256 vb = _mm256_loadu_ps(&inB[blockStart]);
                __m256 vc = opVec(va, vb);
                _mm256_storeu_ps(&outC[blockStart], vc);
            }

#elif defined(SIMD_NEON)
            float32x4_t va = vld1q_f32(&inA[blockStart]);
            float32x4_t vb = vld1q_f32(&inB[blockStart]);
            float32x4_t vc = opVec(va, vb);
            vst1q_f32(&outC[blockStart], vc);

#else
            // fallback
            for(std::size_t i=0; i<chunk; i++){
                float a = inA[blockStart + i];
                float b = inB[blockStart + i];
                outC[blockStart + i] = opScalar(a, b);
            }
#endif
        }
    }
}

//============================================================
// (6) 3バッファ: stride≠1
//============================================================
/**
 * @brief [3バッファ(inA, inB) => outC, stride!=1]
 *        outC[idxC] = f(inA[idxA], inB[idxB])
 */
template<class OpVec, class OpScalar>
void iter_3buf_f32_strN_vec_omp(
    const float* inA, std::size_t strideA,
    const float* inB, std::size_t strideB,
    float*       outC, std::size_t strideC,
    std::size_t  n,
    OpVec        opVec,
    OpScalar     opScalar
)
{
#if defined(SIMD_AVX512)
    const std::size_t step = 16;
#elif defined(SIMD_AVX2)
    const std::size_t step = 8;
#else
    const std::size_t step = 1;
#endif

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for(std::size_t blockStart=0; blockStart < n; blockStart += step)
        {
            std::size_t remain = n - blockStart;
            std::size_t chunk  = (remain>=step)? step : remain;

            if(chunk<step || step==1) {
                // fallback
                for(std::size_t i=0; i<chunk; i++){
                    std::size_t idxA = (blockStart + i)*strideA;
                    std::size_t idxB = (blockStart + i)*strideB;
                    std::size_t idxC = (blockStart + i)*strideC;
                    float a = inA[idxA];
                    float b = inB[idxB];
                    outC[idxC] = opScalar(a, b);
                }
                continue;
            }

#if defined(SIMD_AVX512)
            int idxA_[16], idxB_[16], idxC_[16];
            for(int i=0; i<16; i++){
                idxA_[i] = int((blockStart + i)*strideA);
                idxB_[i] = int((blockStart + i)*strideB);
                idxC_[i] = int((blockStart + i)*strideC);
            }
            // gather
            __m512 va = gather_f32_avx512(inA, idxA_, 16);
            __m512 vb = gather_f32_avx512(inB, idxB_, 16);
            // calc
            __m512 vc = opVec(va, vb);
            // scatter
            __m512i viC = _mm512_loadu_si512((const void*)idxC_);
            _mm512_i32scatter_ps(outC, viC, vc, 4);

#elif defined(SIMD_AVX2)
            int idxA_[8], idxB_[8], idxC_[8];
            for(int i=0; i<8; i++){
                idxA_[i] = int((blockStart + i)*strideA);
                idxB_[i] = int((blockStart + i)*strideB);
                idxC_[i] = int((blockStart + i)*strideC);
            }
            __m256 va = gather_f32_avx2(inA, idxA_, 8);
            __m256 vb = gather_f32_avx2(inB, idxB_, 8);
            __m256 vc = opVec(va, vb);
            // scatter
            float tmp[8];
            _mm256_storeu_ps(tmp, vc);
            for(int i=0; i<8; i++){
                outC[idxC_[i]] = tmp[i];
            }
#endif
        }
    }
}

