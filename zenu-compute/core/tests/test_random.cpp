#include <gtest/gtest.h>
#include <cmath>
#include <vector>

// CUDA runtime ヘッダ (GPUテストで cudaMalloc/cudaMemcpy 等を使う)
#include <cuda_runtime.h>

#include "zenu_compute.h"

// ======================================================
// 1) CPU版: 正規分布 (F32) + seed
// ======================================================
TEST(RandomTest, NormalDistributionF32) {
    const int N = 100000;           // 生成する乱数の個数
    const float mean = 1.0f;        // 期待平均
    const float stddev = 2.0f;      // 期待標準偏差
    const unsigned long long seed = 1234ULL; // シード追加

    std::vector<float> data(N);

    // 関数呼び出し (seedを追加)
    ZenuStatus st = zenu_compute_normal_distribution_cpu(
        data.data(),
        N,
        mean,
        stddev,
        f32,
        seed
    );
    ASSERT_EQ(st, Success);

    // 生成されたデータの平均と標準偏差を簡単にチェック
    double sum = 0.0;
    for (auto &x : data) {
        sum += x;
    }
    double actual_mean = sum / N;

    double sq_sum = 0.0;
    for (auto &x : data) {
        double diff = x - actual_mean;
        sq_sum += diff * diff;
    }
    double actual_std = std::sqrt(sq_sum / (N - 1));

    // 平均が近いことを確認 (誤差は統計的に ±0.1~0.2 程度は許容)
    EXPECT_NEAR(actual_mean, mean, 0.2);

    // 標準偏差が近いことを確認
    EXPECT_NEAR(actual_std, stddev, 0.2);
}

// ======================================================
// 2) CPU版: 正規分布 (F64) + seed
// ======================================================
TEST(RandomTest, NormalDistributionF64) {
    const int N = 100000;
    const double mean = 5.0;
    const double stddev = 0.5;
    const unsigned long long seed = 9999ULL;

    std::vector<double> data(N);

    // f64 で生成 (seed追加)
    ZenuStatus st = zenu_compute_normal_distribution_cpu(
        data.data(),
        N,
        static_cast<float>(mean),
        static_cast<float>(stddev),
        f64,
        seed
    );
    ASSERT_EQ(st, Success);

    // 生成されたデータの平均と標準偏差
    double sum = 0.0;
    for (auto &x : data) {
        sum += x;
    }
    double actual_mean = sum / N;

    double sq_sum = 0.0;
    for (auto &x : data) {
        double diff = x - actual_mean;
        sq_sum += diff * diff;
    }
    double actual_std = std::sqrt(sq_sum / (N - 1));

    EXPECT_NEAR(actual_mean, mean, 0.05);
    EXPECT_NEAR(actual_std, stddev, 0.05);
}

// ======================================================
// 3) CPU版: 一様分布 (F32) + seed
// ======================================================
TEST(RandomTest, UniformDistributionF32) {
    const int N = 100000;
    const float low = -1.0f;
    const float high = 3.0f;
    const unsigned long long seed = 2023ULL;

    std::vector<float> data(N);

    ZenuStatus st = zenu_compute_uniform_distribution_cpu(
        data.data(),
        N,
        low,
        high,
        f32,
        seed
    );
    ASSERT_EQ(st, Success);

    // min, max
    float minVal = data[0];
    float maxVal = data[0];
    double sum = 0.0;
    for (auto &x : data) {
        if (x < minVal) minVal = x;
        if (x > maxVal) maxVal = x;
        sum += x;
    }
    EXPECT_GE(minVal, low);
    EXPECT_LE(maxVal, high);

    // 一様分布 => 理論平均は (low+high)/2
    double actual_mean = sum / N;
    double expected_mean = 0.5 * (low + high);
    EXPECT_NEAR(actual_mean, expected_mean, 0.1 * (high - low)); 
}

// ======================================================
// 4) CPU版: 一様分布 (F64) + seed
// ======================================================
TEST(RandomTest, UniformDistributionF64) {
    const int N = 100000;
    const double low = 100.0;
    const double high = 200.0;
    const unsigned long long seed = 123456ULL;

    std::vector<double> data(N);

    ZenuStatus st = zenu_compute_uniform_distribution_cpu(
        data.data(),
        N,
        static_cast<float>(low),
        static_cast<float>(high),
        f64,
        seed
    );
    ASSERT_EQ(st, Success);

    double minVal = data[0];
    double maxVal = data[0];
    double sum = 0.0;
    for (auto &x : data) {
        if (x < minVal) minVal = x;
        if (x > maxVal) maxVal = x;
        sum += x;
    }
    EXPECT_GE(minVal, low);
    EXPECT_LE(maxVal, high);

    double actual_mean = sum / N;
    double expected_mean = 0.5 * (low + high);
    EXPECT_NEAR(actual_mean, expected_mean, 0.2 * (high - low));
}

// =====================================================================
// 5) GPU版: 正規分布 (F32)
// =====================================================================
TEST(RandomTest, NormalDistributionNvidiaF32)
{
    const int N = 100000;
    const float mean = 2.0f;
    const float stddev = 3.0f;
    const unsigned long long seed = 2345ULL;
    const int device_id = 0; // 使うGPU

    // GPUメモリ確保
    float* d_data = nullptr;
    cudaMalloc((void**)&d_data, N*sizeof(float));
    ASSERT_NE(d_data, nullptr);

    // 生成呼び出し
    ZenuStatus st = zenu_compute_normal_distribution_nvidia(
        d_data,
        N,
        mean,
        stddev,
        f32,
        device_id,
        seed
    );
    ASSERT_EQ(st, Success);

    // 結果をホストに持ち帰る
    std::vector<float> data(N);
    cudaMemcpy(data.data(), d_data, N*sizeof(float), cudaMemcpyDeviceToHost);

    // min, max, mean, std確認
    float minVal = data[0];
    float maxVal = data[0];
    double sum = 0.0;
    for(auto &x : data){
        if(x<minVal) minVal = x;
        if(x>maxVal) maxVal = x;
        sum += x;
    }
    double actual_mean = sum / N;

    double sq_sum = 0.0;
    for(auto &x : data){
        double diff = x - actual_mean;
        sq_sum += diff*diff;
    }
    double actual_std = std::sqrt(sq_sum / (N -1));

    // 期待範囲
    // (ただし統計的な誤差を考慮、±0.2*stddev 程度は許容)
    EXPECT_NEAR(actual_mean, mean, 0.3);
    EXPECT_NEAR(actual_std, stddev, 0.3);

    // 後始末
    cudaFree(d_data);
}

// =====================================================================
// 6) GPU版: 正規分布 (F64)
// =====================================================================
TEST(RandomTest, NormalDistributionNvidiaF64)
{
    const int N = 100000;
    const double mean = 10.0;
    const double stddev = 1.0;
    const unsigned long long seed = 99999ULL;
    const int device_id = 0;

    // GPUメモリ
    double* d_data = nullptr;
    cudaMalloc((void**)&d_data, N*sizeof(double));
    ASSERT_NE(d_data, nullptr);

    // 生成
    ZenuStatus st = zenu_compute_normal_distribution_nvidia(
        d_data,
        N,
        (float)mean,
        (float)stddev,
        f64,
        device_id,
        seed
    );
    ASSERT_EQ(st, Success);

    // copy back
    std::vector<double> data(N);
    cudaMemcpy(data.data(), d_data, N*sizeof(double), cudaMemcpyDeviceToHost);

    // mean, std
    double sum=0.0;
    for(auto &x : data) sum+= x;
    double actual_mean = sum/N;

    double sq_sum=0.0;
    for(auto &x : data){
        double diff = x - actual_mean;
        sq_sum += diff*diff;
    }
    double actual_std = std::sqrt(sq_sum / (N-1));

    EXPECT_NEAR(actual_mean, mean, 0.2);
    EXPECT_NEAR(actual_std, stddev, 0.2);

    cudaFree(d_data);
}

// =====================================================================
// 7) GPU版: 一様分布 (F32)
// =====================================================================
TEST(RandomTest, UniformDistributionNvidiaF32)
{
    const int N= 100000;
    const float low = -2.0f;
    const float high=  4.0f;
    const unsigned long long seed = 1357ULL;
    const int device_id = 0;

    float* d_data=nullptr;
    cudaMalloc((void**)&d_data, N*sizeof(float));
    ASSERT_NE(d_data, nullptr);

    ZenuStatus st = zenu_compute_uniform_distribution_nvidia(
        d_data,
        N,
        low,
        high,
        f32,
        device_id,
        seed
    );
    ASSERT_EQ(st, Success);

    // copy back
    std::vector<float> data(N);
    cudaMemcpy(data.data(), d_data, N*sizeof(float), cudaMemcpyDeviceToHost);

    float minVal = data[0];
    float maxVal = data[0];
    double sum=0.0;
    for(auto &x: data){
        if(x<minVal) minVal=x;
        if(x>maxVal) maxVal=x;
        sum+=x;
    }
    EXPECT_GE(minVal, low);
    EXPECT_LE(maxVal, high);

    double actual_mean = sum/N;
    double expected_mean = 0.5*(low+high);
    // 許容誤差: ±0.1*(range)
    double range = (high-low);
    EXPECT_NEAR(actual_mean, expected_mean, 0.1* range);

    cudaFree(d_data);
}

// =====================================================================
// 8) GPU版: 一様分布 (F64)
// =====================================================================
TEST(RandomTest, UniformDistributionNvidiaF64)
{
    const int N= 100000;
    const double low= 50.0;
    const double high= 60.0;
    const unsigned long long seed = 2468ULL;
    const int device_id=0;

    double* d_data=nullptr;
    cudaMalloc((void**)&d_data, N*sizeof(double));
    ASSERT_NE(d_data, nullptr);

    ZenuStatus st = zenu_compute_uniform_distribution_nvidia(
        d_data,
        N,
        (float)low,
        (float)high,
        f64,
        device_id,
        seed
    );
    ASSERT_EQ(st, Success);

    std::vector<double> data(N);
    cudaMemcpy(data.data(), d_data, N*sizeof(double), cudaMemcpyDeviceToHost);

    double minVal=data[0];
    double maxVal=data[0];
    double sum=0.0;
    for(auto &x: data){
        if(x<minVal) minVal=x;
        if(x>maxVal) maxVal=x;
        sum+=x;
    }
    EXPECT_GE(minVal, low);
    EXPECT_LE(maxVal, high);

    double actual_mean= sum/N;
    double expected_mean= 0.5*(low+high);
    double range=(high-low);
    EXPECT_NEAR(actual_mean, expected_mean, 0.1* range);

    cudaFree(d_data);
}

// main 関数 (GoogleTest が用意しているマクロ)
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

