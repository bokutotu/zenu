#include "zenu_compute.h"

#include <random>
#include <cmath>
#include <limits>
#include <omp.h>

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h> // NEON
#endif

#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h> // AVX / AVX2
#endif

//===========================================================
// スレッドごとに独立したエンジンを持たせるための構造体
//===========================================================
typedef struct {
    std::mt19937_64 rng;
} ThreadLocalRNG;

// OpenMP スレッドごとの乱数エンジンを保持する (実際には thread_local でも可)
static ThreadLocalRNG* g_rng_array = nullptr;
static int g_max_threads = 0;

//===========================================================
// スレッドごとの RNG を初期化
//===========================================================
static void init_rng_thread_local()
{
    int nThreads = omp_get_max_threads();
    if (nThreads <= 0) { nThreads = 1; }

    // 既に確保済みの場合は再利用(単純のためここでは再確保)
    if (g_max_threads < nThreads) {
        if (g_rng_array) {
            delete[] g_rng_array;
            g_rng_array = nullptr;
        }
        g_rng_array = new ThreadLocalRNG[nThreads];
        g_max_threads = nThreads;
    }

    // seed の割り当て（簡易的に）
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
    return std::generate_canonical<double, std::numeric_limits<double>::digits>(rng);
}

//===========================================================
// Box-Muller 法 (スカラー版) 2つの正規乱数を返す
// z0 = r*cos(theta), z1 = r*sin(theta)
//===========================================================
static inline void box_muller_2(double u1, double u2, double &z0, double &z1)
{
    double r = std::sqrt(-2.0 * std::log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;
    z0 = r * std::cos(theta);
    z1 = r * std::sin(theta);
}

//===========================================================
// NEON 実装 (float32x4_t で4要素まとめて)
// - 1回の実行で "4ペア" → 8 個の正規乱数を生成したほうがベストだが、
//   ここでは簡単のため "2ペア" (4 個の正規乱数)だけ示す例
//===========================================================
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
static inline void box_muller_neon_f32(float *dst, std::mt19937_64 &rng)
{
    // u1, u2 を4要素生成
    float u1tmp[4], u2tmp[4];
    for (int i = 0; i < 4; i++) {
        u1tmp[i] = static_cast<float>(rand_0_1(rng));
        u2tmp[i] = static_cast<float>(rand_0_1(rng));
    }
    float32x4_t u1 = vld1q_f32(u1tmp);
    float32x4_t u2 = vld1q_f32(u2tmp);

    // r = sqrt(-2 log(u1))
    float32x4_t neg2 = vdupq_n_f32(-2.0f);
    float32x4_t log_u1 = vlogq_f32(u1);      // NEON拡張: vlogq_f32
    float32x4_t r      = vmulq_f32(neg2, log_u1);
    r = vsqrtq_f32(r);

    // theta = 2π * u2
    float32x4_t two_pi = vdupq_n_f32(6.283185307f);
    float32x4_t theta  = vmulq_f32(two_pi, u2);

    // z0 = r*cos(theta), z1 = r*sin(theta)
    float32x4_t z0 = vmulq_f32(r, vcosq_f32(theta)); // NEON拡張: vcosq_f32
    float32x4_t z1 = vmulq_f32(r, vsinq_f32(theta)); // vsinq_f32

    // 4要素: z0[0], z1[0], z0[1], z1[1], ... と交互に書き出す例
    float tmpZ0[4], tmpZ1[4];
    vst1q_f32(tmpZ0, z0);
    vst1q_f32(tmpZ1, z1);

    for (int i = 0; i < 4; i++) {
        dst[2*i + 0] = tmpZ0[i];
        dst[2*i + 1] = tmpZ1[i];
    }
}
#endif

//===========================================================
// AVX 実装 (float, 8要素) の簡易例
// - AVX/AVX2で vlog_ps, vcos_ps, vsin_ps はコンパイラによっては
//   intrinsic が無いので、手書き実装か __builtin_prototypes の利用が必要
//   ここでは簡単のために _mm256_log_ps, _mm256_cos_ps, ... がある前提で書く
//   (実際には自前実装か SVML/短縮近似を利用)
//===========================================================
#if defined(__AVX__) || defined(__AVX2__)
static inline void box_muller_avx_f32(float *dst, std::mt19937_64 &rng)
{
    // 8個の (u1, u2) を用意
    alignas(32) float u1tmp[8], u2tmp[8];
    for (int i = 0; i < 8; i++) {
        u1tmp[i] = static_cast<float>(rand_0_1(rng));
        u2tmp[i] = static_cast<float>(rand_0_1(rng));
    }
    __m256 u1 = _mm256_load_ps(u1tmp);
    __m256 u2 = _mm256_load_ps(u2tmp);

    // r = sqrt(-2 log(u1))
    __m256 neg2 = _mm256_set1_ps(-2.0f);
    __m256 r    = _mm256_mul_ps(neg2, _mm256_log_ps(u1)); 
    r = _mm256_sqrt_ps(r);

    // theta = 2π * u2
    __m256 two_pi = _mm256_set1_ps(6.283185307f);
    __m256 theta  = _mm256_mul_ps(two_pi, u2);

    // z0 = r*cos(theta), z1 = r*sin(theta)
    __m256 z0 = _mm256_mul_ps(r, _mm256_cos_ps(theta));
    __m256 z1 = _mm256_mul_ps(r, _mm256_sin_ps(theta));

    // z0, z1 を交互に出力 (8要素ずつ)
    // 例: out[0] = z0[0], out[1] = z1[0], ...
    alignas(32) float tmpZ0[8], tmpZ1[8];
    _mm256_store_ps(tmpZ0, z0);
    _mm256_store_ps(tmpZ1, z1);

    for (int i = 0; i < 8; i++) {
        dst[2*i + 0] = tmpZ0[i];
        dst[2*i + 1] = tmpZ1[i];
    }
}
#endif

//===========================================================
// メイン関数: CPU実装で正規分布を生成
//===========================================================
ZenuStatus zenu_compute_normal_distribution_cpu(void* dst,
                                                int num_elements,
                                                float mean,
                                                float stddev,
                                                ZenuDataType data_type)
{
    if (!dst) {
        return InvalidArgument;
    }
    if (num_elements < 0) {
        return InvalidArgument;
    }
    if (stddev < 0.0f) {
        return InvalidArgument;
    }

    // RNG 初期化
    init_rng_thread_local();

    // スレッド並列で処理
    // Box-Muller 1回で 2要素出るので、ループは (i += 2) のイメージ
    // SIMD実装では一度にもっと多く出力するが、最終的には
    // 生成総数が num_elements に一致するように制御すれば良い。
    if (data_type == f32)
    {
        float* out_f32 = static_cast<float*>(dst);

        // (mean, stddev) をベクトル結果に加算・乗算
        //   SIMD生成した z を z*stddev + mean に変換
        //   => 後段でまとめて実施

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::mt19937_64& rng = g_rng_array[tid].rng;

#pragma omp for schedule(static)
            for (int i = 0; i < num_elements; i += 2)
            {
                // まず2つの正規乱数(z0,z1)を生成 (SIMD or スカラー)

                float z0, z1; 
#if defined(__AVX2__) || (defined(__AVX__) && !defined(__AVX2__))
                // AVX / AVX2を想定: 一気に8個生成して、そのうち2つを使う (簡易)
                // 実際には一度に 8 or 16個をバッファに貯めこんで i の端数まで再利用するほうが効率良い
                // ここでは簡単のために「毎回 box_muller_avx_f32 を呼んで 16個中2個しか使わない」というのは非効率
                // → サンプルなので割愛

                float tmp[16]; // 8 (u1,u2) → 16 出力
                box_muller_avx_f32(tmp, rng);
                // tmp[0..15] に 16個の正規乱数が格納 (z0,z1,z0,z1,... 8組)
                // とりあえず先頭2つを使う例
                z0 = tmp[0];
                z1 = tmp[1];

#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
                // NEON: 一度に4要素(=2ペア)生成 → 出力先頭2つを使用
                float tmp[8]; // 4要素→実質 z0[4],z1[4]で8個
                box_muller_neon_f32(tmp, rng);
                z0 = tmp[0];
                z1 = tmp[1];

#else
                // スカラーフォールバック
                // 2つ同時に生成
                double u1 = rand_0_1(rng);
                double u2 = rand_0_1(rng);
                double bm0, bm1;
                box_muller_2(u1, u2, bm0, bm1);
                z0 = static_cast<float>(bm0);
                z1 = static_cast<float>(bm1);
#endif
                // z0,z1 を dst に書き込み (端数チェック)
                // (z0,z1)*stddev + mean
                out_f32[i] = z0 * stddev + mean;
                if (i+1 < num_elements) {
                    out_f32[i+1] = z1 * stddev + mean;
                }
            }
        } // omp parallel
    }
    else  // f64
    {
        double* out_f64 = static_cast<double*>(dst);
        double dmean = static_cast<double>(mean);
        double dstd  = static_cast<double>(stddev);

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::mt19937_64& rng = g_rng_array[tid].rng;

#pragma omp for schedule(static)
            for (int i = 0; i < num_elements; i += 2)
            {
                // f64の場合は、NEONでdoubleをベクトル化するのは
                // 一部しかサポートされず手間が大きいので、ここではスカラー実装
                // AVX/AVX2は __m256d を使えば4要素同時にいけるが、ここでは簡易例
                // TODO: 上記

                double u1 = rand_0_1(rng);
                double u2 = rand_0_1(rng);
                double z0, z1;
                box_muller_2(u1, u2, z0, z1);

                out_f64[i] = z0 * dstd + dmean;
                if (i + 1 < num_elements) {
                    out_f64[i + 1] = z1 * dstd + dmean;
                }
            }
        } // omp parallel
    }

    return Success;
}

