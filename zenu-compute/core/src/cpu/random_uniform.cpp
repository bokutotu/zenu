#include "zenu_compute.h"

#include <random>
#include <cmath>
#include <limits>
#include <omp.h>

// For NEON
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// For AVX / AVX2
#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif

//===========================================================
// スレッドごとに独立した RNG を持たせる (衝突防止・並列性能向上のため)
//===========================================================
typedef struct {
    std::mt19937_64 rng;
} ThreadLocalRNG;

static ThreadLocalRNG* g_rng_array = nullptr;
static int g_max_threads = 0;

//===========================================================
// スレッドごとの RNG 初期化
//===========================================================
static void init_rng_thread_local()
{
    int nThreads = omp_get_max_threads();
    if (nThreads <= 0) { nThreads = 1; }

    // 既に確保済みの場合は再確保する(簡易実装)
    if (g_max_threads < nThreads) {
        if (g_rng_array) {
            delete[] g_rng_array;
            g_rng_array = nullptr;
        }
        g_rng_array = new ThreadLocalRNG[nThreads];
        g_max_threads = nThreads;
    }

    // 各スレッドの rng に異なる seed を設定
    std::random_device rd;
    for (int i = 0; i < nThreads; i++) {
        uint64_t seed_val = (static_cast<uint64_t>(rd()) << 32) ^ rd();
        g_rng_array[i].rng.seed(seed_val);
    }
}

//===========================================================
// [0,1) の一様乱数を double で生成 (スカラー)
//===========================================================
static inline double rand_0_1(std::mt19937_64 &rng)
{
    // generate_canonical は [0,1) の double
    return std::generate_canonical<double, std::numeric_limits<double>::digits>(rng);
}

//===========================================================
// f32用: NEONで一度に4個の [0,1) を生成 → (4要素 float32x4_t)
//        ここでは "4要素分" の乱数を生成してレジスタにロードする例
//===========================================================
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
static inline float32x4_t neon_rand_0_1_4(std::mt19937_64 &rng)
{
    float tmp[4];
    for (int i = 0; i < 4; i++) {
        tmp[i] = static_cast<float>(rand_0_1(rng)); // [0,1)
    }
    return vld1q_f32(tmp); // 4要素一気にロード
}
#endif

//===========================================================
// f32用: AVX/AVX2で一度に8個の [0,1) を生成 → (__m256)
//===========================================================
#if defined(__AVX__) || defined(__AVX2__)
static inline __m256 avx_rand_0_1_8(std::mt19937_64 &rng)
{
    alignas(32) float tmp[8];
    for (int i = 0; i < 8; i++) {
        tmp[i] = static_cast<float>(rand_0_1(rng));
    }
    return _mm256_load_ps(tmp);
}
#endif

//===========================================================
// メイン実装: CPUで一様分布を生成
//===========================================================
ZenuStatus zenu_compute_uniform_distribution_cpu(void* dst,
                                                 int num_elements,
                                                 float low,
                                                 float high,
                                                 ZenuDataType data_type)
{
    if (!dst) {
        return InvalidArgument;
    }
    if (num_elements < 0) {
        return InvalidArgument;
    }
    if (low > high) {
        return InvalidArgument;
    }

    // RNG 初期化
    init_rng_thread_local();

    // range = high - low
    double range = static_cast<double>(high) - static_cast<double>(low);

    //========================================
    // case: float (f32)
    //========================================
    if (data_type == f32)
    {
        float* out_f32 = static_cast<float*>(dst);

        // OpenMP 並列
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::mt19937_64& rng = g_rng_array[tid].rng;

            // SIMD化: AVX / NEON / スカラー で分岐
#if defined(__AVX2__) || defined(__AVX__)
#pragma omp for schedule(static)
            for (int i = 0; i < num_elements; i += 8)
            {
                // __m256 rand0_1 = 8要素
                __m256 r = avx_rand_0_1_8(rng); // [0,1) x 8
                // multiply by range, add low
                __m256 range_v = _mm256_set1_ps((float)range);
                __m256 low_v   = _mm256_set1_ps(low);

                // out = r*range + low
                __m256 vals = _mm256_fmadd_ps(r, range_v, low_v); 
                // 端数処理: i+7 までOKかチェック
                if (i + 8 <= num_elements) {
                    _mm256_storeu_ps(&out_f32[i], vals);
                }
                else {
                    // 端数 (num_elements % 8 != 0) の場合
                    alignas(32) float tmp[8];
                    _mm256_store_ps(tmp, vals);
                    int remain = num_elements - i;
                    for (int k = 0; k < remain; k++) {
                        out_f32[i + k] = tmp[k];
                    }
                }
            }

#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
#pragma omp for schedule(static)
            for (int i = 0; i < num_elements; i += 4)
            {
                // float32x4_t r = [0,1) x 4
                float32x4_t r = neon_rand_0_1_4(rng);
                float32x4_t range_v = vdupq_n_f32((float)range);
                float32x4_t low_v   = vdupq_n_f32(low);

                // out = r*range + low
                float32x4_t vals = vmlaq_f32(low_v, r, range_v);

                // 端数チェック
                if (i + 4 <= num_elements) {
                    vst1q_f32(&out_f32[i], vals);
                }
                else {
                    float tmp[4];
                    vst1q_f32(tmp, vals);
                    int remain = num_elements - i;
                    for (int k = 0; k < remain; k++) {
                        out_f32[i + k] = tmp[k];
                    }
                }
            }

#else
            // スカラーフォールバック
#pragma omp for schedule(static)
            for (int i = 0; i < num_elements; i++)
            {
                double r = rand_0_1(rng); // [0,1)
                double val = r * range + (double)low;
                out_f32[i] = static_cast<float>(val);
            }
#endif
        } // omp parallel
    }
    //========================================
    // case: double (f64)
    //========================================
    else
    {
        double* out_f64 = static_cast<double*>(dst);

        // OpenMP 並列
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::mt19937_64& rng = g_rng_array[tid].rng;

#if defined(__AVX2__) || defined(__AVX__)
            // 参考: AVX なら __m256d で4要素同時にできる
            // ただし __m256d に対して標準関数(log, cos など)はSVML依存 or 自前実装になる
            // 今回は一様分布なので x = [rand_0_1]*range + low をするだけ
#pragma omp for schedule(static)
            for (int i = 0; i < num_elements; i += 4)
            {
                alignas(32) double tmp[4];
                for (int j = 0; j < 4; j++) {
                    tmp[j] = rand_0_1(rng) * range + (double)low;
                }
                // SIMDロード/ストア (一応記述例)
                __m256d vals = _mm256_load_pd(tmp);
                if (i + 4 <= num_elements) {
                    _mm256_storeu_pd(&out_f64[i], vals);
                } else {
                    int remain = num_elements - i;
                    _mm256_store_pd(tmp, vals);
                    for (int k = 0; k < remain; k++) {
                        out_f64[i + k] = tmp[k];
                    }
                }
            }

#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
            // NEON で double をベクトル化できるかは限定的 (Armv8以降2レーン)
            // ここでは簡単のためスカラーフォールバック
#pragma omp for schedule(static)
            for (int i = 0; i < num_elements; i++)
            {
                double r = rand_0_1(rng);
                out_f64[i] = r * range + (double)low;
            }

#else
            // スカラーフォールバック (x86-SSE, etc.)
#pragma omp for schedule(static)
            for (int i = 0; i < num_elements; i++)
            {
                double r = rand_0_1(rng);
                out_f64[i] = r * range + (double)low;
            }
#endif
        } // omp parallel
    }

    return Success;
}

