#pragma once

#include <cstddef>
#include <cstdint> // for uintptr_t
#include <omp.h>

/**
 * @brief アドレスが指定した align バイト境界に乗っているかを判定する
 */
static inline bool is_aligned_f64(const void* ptr, std::size_t align)
{
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    return (addr % align) == 0;
}

//============================================================
// SIMD 判定 (倍精度も同じマクロで判定)
//============================================================
#if defined(__AVX512F__)
#  include <immintrin.h>
#  define SIMD_AVX512_F64
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define SIMD_AVX2_F64
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
// NEON doubleは一部アーキで float64x2_t (2要素) などがあるが、
// ここではシンプルに fallback にするか、必要に応じて追加
#  define SIMD_NEON_F64
#else
#  define SIMD_NONE_F64
#endif

//============================================================
// AVX2/AVX512用 gather ユーティリティ (double, stride≠1)
//============================================================
#if defined(SIMD_AVX512_F64) || defined(SIMD_AVX2_F64)
/**
 * @brief gather_f64_avx2: AVX2(256-bit)用の double版 gather (4要素)
 *
 * @param base     ベースポインタ(double配列)
 * @param indices  インデックス配列(要素数4)
 * @return __m256d
 */
static inline __m256d gather_f64_avx2(const double* base, const int* indices, int count=4)
{
    __m128i vi = _mm_loadu_si128((const __m128i*)indices); 
    // AVX2: _mm256_i32gather_pd(base, vi, 8) が使える (8=sizeof(double))
    // ただし要GCC 4.8+ etc.
    return _mm256_i32gather_pd(base, vi, 8);
}

#if defined(SIMD_AVX512_F64)
/**
 * @brief gather_f64_avx512: AVX512(512-bit)用の double版 gather (8要素)
 */
static inline __m512d gather_f64_avx512(const double* base, const int* indices, int count=8)
{
    __m256i vi = _mm256_loadu_si256((const __m256i*)indices);
    // 8=sizeof(double)
    return _mm512_i32gather_pd(vi, base, 8);
}
#endif
#endif // gather double

//============================================================
// 1バッファ: stride=1 (double版)
//============================================================
/**
 * @brief [1バッファ, stride=1, double版]
 *   buf[i] = f(buf[i])
 *
 * @tparam OpVecOne    ex: __m256d operator()(__m256d)
 * @tparam OpScalarOne ex: double operator()(double)
 */
template<class OpVecOne, class OpScalarOne>
void iter_1buf_f64_str1_vec_omp(
    double* buf,
    std::size_t n,
    OpVecOne   opVec,
    OpScalarOne opScalar)
{
#if defined(SIMD_AVX512_F64)
    constexpr std::size_t alignment = 64; 
    constexpr std::size_t step = 8;   // AVX512 double: 512bit = 8要素(double)
#elif defined(SIMD_AVX2_F64)
    constexpr std::size_t alignment = 32;
    constexpr std::size_t step = 4;   // AVX2 double: 256bit = 4要素(double)
#else
    // NEON double => fallback, or 2要素...
    constexpr std::size_t alignment = 16;
    constexpr std::size_t step = 1;
#endif

    bool aligned_head = is_aligned_f64(buf, alignment);

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for(std::size_t blockStart=0; blockStart<n; blockStart+=step)
        {
            std::size_t remain = n - blockStart;
            std::size_t chunk  = (remain>=step ? step : remain);

            if(chunk<step) {
                // fallback scalar
                for(std::size_t i=0;i<chunk;i++){
                    buf[blockStart + i] = opScalar(buf[blockStart + i]);
                }
                continue;
            }

#if defined(SIMD_AVX512_F64)
            if(aligned_head) {
                __m512d x = _mm512_load_pd(&buf[blockStart]);  // aligned load
                x = opVec(x);
                _mm512_store_pd(&buf[blockStart], x);
            } else {
                __m512d x = _mm512_loadu_pd(&buf[blockStart]); // unaligned
                x = opVec(x);
                _mm512_storeu_pd(&buf[blockStart], x);
            }

#elif defined(SIMD_AVX2_F64)
            if(aligned_head) {
                __m256d x = _mm256_load_pd(&buf[blockStart]);
                x = opVec(x);
                _mm256_store_pd(&buf[blockStart], x);
            } else {
                __m256d x = _mm256_loadu_pd(&buf[blockStart]);
                x = opVec(x);
                _mm256_storeu_pd(&buf[blockStart], x);
            }

#else
            // fallback
            for(std::size_t i=0;i<chunk;i++){
                buf[blockStart + i] = opScalar(buf[blockStart + i]);
            }
#endif
        }
    }
}

//============================================================
// 1バッファ: stride≠1 (double版)
//============================================================
/**
 * @brief [1バッファ, stride!=1, double版]
 *   buf[i*stride] = f(buf[i*stride]), gather/scatter
 */
template<class OpVecOne, class OpScalarOne>
void iter_1buf_f64_strN_vec_omp(
    double* buf,
    std::size_t n,
    std::size_t stride,
    OpVecOne   opVec,
    OpScalarOne opScalar)
{
#if defined(SIMD_AVX512_F64)
    const std::size_t step = 8;  // 8要素(double)
#elif defined(SIMD_AVX2_F64)
    const std::size_t step = 4;  // 4要素(double)
#else
    const std::size_t step = 1;
#endif

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for(std::size_t blockStart=0; blockStart<n; blockStart+=step)
        {
            std::size_t remain = n - blockStart;
            if(remain>=step && step>1)
            {
#if defined(SIMD_AVX512_F64)
                int idx[8];
                for(int i=0;i<8;i++){
                    idx[i] = int((blockStart + i)*stride);
                }
                __m512d x = gather_f64_avx512(buf, idx, 8);
                x = opVec(x);
                // scatter
                __m256i vid = _mm256_loadu_si256((const __m256i*)idx);
                _mm512_i32scatter_pd(buf, vid, x, 8);

#elif defined(SIMD_AVX2_F64)
                int idx[4];
                for(int i=0;i<4;i++){
                    idx[i] = int((blockStart + i)*stride);
                }
                __m256d x = gather_f64_avx2(buf, idx, 4);
                x = opVec(x);
                // scatter
                double tmp[4];
                _mm256_storeu_pd(tmp, x);
                for(int i=0;i<4;i++){
                    buf[idx[i]] = tmp[i];
                }
#endif
            }
            else {
                // fallback
                std::size_t actual = (remain<step)? remain : step;
                for(std::size_t i=0;i<actual;i++){
                    std::size_t pos = (blockStart + i)*stride;
                    buf[pos] = opScalar(buf[pos]);
                }
            }
        }
    }
}

//============================================================
// 2バッファ: stride=1 (double版)
//============================================================
/**
 * @brief [2バッファ(inA, outB), double, stride=1]
 *   outB[i] = opScalar(inA[i], outB[i])
 */
template<class OpVec, class OpScalar>
void iter_2buf_f64_str1_vec_omp(
    const double* inA,
    double*       outB,
    std::size_t   n,
    OpVec         opVec,     ///< ex: __m256d operator()(__m256d, __m256d)
    OpScalar      opScalar   ///< ex: double operator()(double, double)
)
{
#if defined(SIMD_AVX512_F64)
    constexpr std::size_t alignment = 64;
    constexpr std::size_t step = 8;  // AVX512 double=8要素
#elif defined(SIMD_AVX2_F64)
    constexpr std::size_t alignment = 32;
    constexpr std::size_t step = 4;  // AVX2 double=4要素
#else
    constexpr std::size_t alignment = 16; // fallback
    constexpr std::size_t step = 1;
#endif

    bool alignedA = is_aligned_f64(inA,  alignment);
    bool alignedB = is_aligned_f64(outB, alignment);
    bool do_aligned = (alignedA && alignedB);

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for(std::size_t blockStart=0; blockStart<n; blockStart+=step)
        {
            std::size_t remain = n - blockStart;
            std::size_t chunk  = (remain>=step)? step : remain;

            if(chunk<step) {
                // scalar fallback
                for(std::size_t i=0;i<chunk;i++){
                    double a = inA[blockStart + i];
                    double b = outB[blockStart + i];
                    outB[blockStart + i] = opScalar(a, b);
                }
                continue;
            }

#if defined(SIMD_AVX512_F64)
            if(do_aligned) {
                __m512d va = _mm512_load_pd(&inA[blockStart]);
                __m512d vb = _mm512_load_pd(&outB[blockStart]);
                __m512d vb_new = opVec(va,vb);
                _mm512_store_pd(&outB[blockStart], vb_new);
            } else {
                __m512d va = _mm512_loadu_pd(&inA[blockStart]);
                __m512d vb = _mm512_loadu_pd(&outB[blockStart]);
                __m512d vb_new = opVec(va, vb);
                _mm512_storeu_pd(&outB[blockStart], vb_new);
            }

#elif defined(SIMD_AVX2_F64)
            if(do_aligned) {
                __m256d va = _mm256_load_pd(&inA[blockStart]);
                __m256d vb = _mm256_load_pd(&outB[blockStart]);
                __m256d vb_new = opVec(va, vb);
                _mm256_store_pd(&outB[blockStart], vb_new);
            } else {
                __m256d va = _mm256_loadu_pd(&inA[blockStart]);
                __m256d vb = _mm256_loadu_pd(&outB[blockStart]);
                __m256d vb_new = opVec(va, vb);
                _mm256_storeu_pd(&outB[blockStart], vb_new);
            }

#else
            // fallback
            for(std::size_t i=0;i<chunk;i++){
                double a = inA[blockStart + i];
                double b = outB[blockStart + i];
                outB[blockStart + i] = opScalar(a, b);
            }
#endif
        }
    }
}

//============================================================
// 2バッファ: stride≠1 (double版)
//============================================================
/**
 * @brief [2バッファ(inA, outB), double, stride!=1]
 *   outB[idxB] = opScalar(inA[idxA], outB[idxB])
 */
template<class OpVec, class OpScalar>
void iter_2buf_f64_strN_vec_omp(
    const double* inA, std::size_t strideA,
    double*       outB, std::size_t strideB,
    std::size_t   n,
    OpVec         opVec,
    OpScalar      opScalar
)
{
#if defined(SIMD_AVX512_F64)
    constexpr std::size_t step = 8;  // 8要素(double)
#elif defined(SIMD_AVX2_F64)
    constexpr std::size_t step = 4;  // 4要素(double)
#else
    constexpr std::size_t step = 1;
#endif

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for(std::size_t blockStart=0; blockStart<n; blockStart+=step)
        {
            std::size_t remain = n - blockStart;
            std::size_t chunk  = (remain>=step)? step : remain;
            if(chunk<step || step==1) {
                // fallback
                for(std::size_t i=0;i<chunk;i++){
                    std::size_t idxA = (blockStart + i)*strideA;
                    std::size_t idxB = (blockStart + i)*strideB;
                    double a = inA[idxA];
                    double b = outB[idxB];
                    outB[idxB] = opScalar(a,b);
                }
                continue;
            }

#if defined(SIMD_AVX512_F64)
            int idxA_[8], idxB_[8];
            for(int i=0;i<8;i++){
                idxA_[i] = int((blockStart + i)*strideA);
                idxB_[i] = int((blockStart + i)*strideB);
            }
            __m512d va = gather_f64_avx512(inA,  idxA_, 8);
            __m512d vb = gather_f64_avx512(outB, idxB_, 8);
            __m512d vb_new = opVec(va, vb);
            // scatter
            __m256i viB = _mm256_loadu_si256((const __m256i*)idxB_);
            _mm512_i32scatter_pd(outB, viB, vb_new, 8);

#elif defined(SIMD_AVX2_F64)
            int idxA_[4], idxB_[4];
            for(int i=0;i<4;i++){
                idxA_[i] = int((blockStart + i)*strideA);
                idxB_[i] = int((blockStart + i)*strideB);
            }
            __m256d va = gather_f64_avx2(inA, idxA_, 4);
            __m256d vb = gather_f64_avx2(outB, idxB_, 4);
            __m256d vb_new = opVec(va, vb);
            double tmp[4];
            _mm256_storeu_pd(tmp, vb_new);
            for(int i=0;i<4;i++){
                outB[idxB_[i]] = tmp[i];
            }
#endif
        }
    }
}

//============================================================
// 3バッファ: stride=1 (double版)
//============================================================
/**
 * @brief [3バッファ(inA, inB) => outC, double, stride=1]
 *   outC[i] = opScalar(inA[i], inB[i])
 */
template<class OpVec, class OpScalar>
void iter_3buf_f64_str1_vec_omp(
    const double* inA,
    const double* inB,
    double*       outC,
    std::size_t   n,
    OpVec         opVec,    ///< ex: __m512d operator()(__m512d, __m512d)
    OpScalar      opScalar  ///< ex: double operator()(double,double)
)
{
#if defined(SIMD_AVX512_F64)
    constexpr std::size_t alignment = 64;
    constexpr std::size_t step = 8; // double: 512bit=8要素
#elif defined(SIMD_AVX2_F64)
    constexpr std::size_t alignment = 32;
    constexpr std::size_t step = 4; // double: 256bit=4要素
#else
    constexpr std::size_t alignment = 16;
    constexpr std::size_t step = 1;
#endif

    bool alignedA = is_aligned_f64(inA,  alignment);
    bool alignedB = is_aligned_f64(inB,  alignment);
    bool alignedC = is_aligned_f64(outC, alignment);
    bool do_aligned = (alignedA && alignedB && alignedC);

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for(std::size_t blockStart=0; blockStart<n; blockStart+=step)
        {
            std::size_t remain = n - blockStart;
            std::size_t chunk  = (remain>=step)? step : remain;
            if(chunk<step) {
                // fallback
                for(std::size_t i=0; i<chunk; i++){
                    double a = inA[blockStart + i];
                    double b = inB[blockStart + i];
                    outC[blockStart + i] = opScalar(a,b);
                }
                continue;
            }

#if defined(SIMD_AVX512_F64)
            if(do_aligned) {
                __m512d va = _mm512_load_pd(&inA[blockStart]);
                __m512d vb = _mm512_load_pd(&inB[blockStart]);
                __m512d vc = opVec(va,vb);
                _mm512_store_pd(&outC[blockStart], vc);
            } else {
                __m512d va = _mm512_loadu_pd(&inA[blockStart]);
                __m512d vb = _mm512_loadu_pd(&inB[blockStart]);
                __m512d vc = opVec(va,vb);
                _mm512_storeu_pd(&outC[blockStart], vc);
            }

#elif defined(SIMD_AVX2_F64)
            if(do_aligned){
                __m256d va = _mm256_load_pd(&inA[blockStart]);
                __m256d vb = _mm256_load_pd(&inB[blockStart]);
                __m256d vc = opVec(va,vb);
                _mm256_store_pd(&outC[blockStart], vc);
            } else {
                __m256d va = _mm256_loadu_pd(&inA[blockStart]);
                __m256d vb = _mm256_loadu_pd(&inB[blockStart]);
                __m256d vc = opVec(va,vb);
                _mm256_storeu_pd(&outC[blockStart], vc);
            }

#else
            // fallback
            for(std::size_t i=0;i<chunk;i++){
                double a = inA[blockStart + i];
                double b = inB[blockStart + i];
                outC[blockStart + i] = opScalar(a,b);
            }
#endif
        }
    }
}

//============================================================
// 3バッファ: stride≠1 (double版)
//============================================================
/**
 * @brief [3バッファ(inA,inB) => outC, double, stride!=1]
 *   outC[idxC] = opScalar(inA[idxA], inB[idxB])
 */
template<class OpVec, class OpScalar>
void iter_3buf_f64_strN_vec_omp(
    const double* inA, std::size_t strideA,
    const double* inB, std::size_t strideB,
    double*       outC, std::size_t strideC,
    std::size_t   n,
    OpVec         opVec,
    OpScalar      opScalar
)
{
#if defined(SIMD_AVX512_F64)
    constexpr std::size_t step = 8;  // double=8要素(512bit)
#elif defined(SIMD_AVX2_F64)
    constexpr std::size_t step = 4;  // double=4要素(256bit)
#else
    constexpr std::size_t step = 1;
#endif

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for(std::size_t blockStart=0; blockStart<n; blockStart+=step)
        {
            std::size_t remain = n - blockStart;
            std::size_t chunk  = (remain>=step)? step : remain;
            if(chunk<step || step==1){
                // fallback
                for(std::size_t i=0;i<chunk;i++){
                    std::size_t idxA = (blockStart + i)*strideA;
                    std::size_t idxB = (blockStart + i)*strideB;
                    std::size_t idxC = (blockStart + i)*strideC;
                    double a = inA[idxA];
                    double b = inB[idxB];
                    outC[idxC] = opScalar(a,b);
                }
                continue;
            }

#if defined(SIMD_AVX512_F64)
            int idxA_[8], idxB_[8], idxC_[8];
            for(int i=0;i<8;i++){
                idxA_[i] = int((blockStart + i)*strideA);
                idxB_[i] = int((blockStart + i)*strideB);
                idxC_[i] = int((blockStart + i)*strideC);
            }
            // gather
            __m512d va = gather_f64_avx512(inA, idxA_, 8);
            __m512d vb = gather_f64_avx512(inB, idxB_, 8);
            __m512d vc = opVec(va, vb);
            // scatter
            __m256i vidxC = _mm256_loadu_si256((const __m256i*)idxC_);
            _mm512_i32scatter_pd(outC, vidxC, vc, 8);

#elif defined(SIMD_AVX2_F64)
            int idxA_[4], idxB_[4], idxC_[4];
            for(int i=0;i<4;i++){
                idxA_[i] = int((blockStart + i)*strideA);
                idxB_[i] = int((blockStart + i)*strideB);
                idxC_[i] = int((blockStart + i)*strideC);
            }
            __m256d va = gather_f64_avx2(inA, idxA_, 4);
            __m256d vb = gather_f64_avx2(inB, idxB_, 4);
            __m256d vc = opVec(va, vb);
            // scatter
            double tmp[4];
            _mm256_storeu_pd(tmp, vc);
            for(int i=0;i<4;i++){
                outC[idxC_[i]] = tmp[i];
            }
#endif
        }
    }
}

