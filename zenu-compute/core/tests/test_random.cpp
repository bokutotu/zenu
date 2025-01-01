#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include "zenu_compute.h"

// ------------------------------
// テスト: 正規分布
// ------------------------------
TEST(RandomTest, NormalDistributionF32) {
    const int N = 100000;           // 生成する乱数の個数
    const float mean = 1.0f;        // 期待平均
    const float stddev = 2.0f;      // 期待標準偏差
    std::vector<float> data(N);

    // 関数呼び出し
    ZenuStatus st = zenu_compute_normal_distribution_cpu(
        data.data(),
        N,
        mean,
        stddev,
        f32
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
    double actual_std = std::sqrt(sq_sum / (N-1));

    // 平均が近いことを確認 (誤差は統計的に ±0.1~0.2 程度は許容)
    EXPECT_NEAR(actual_mean, mean, 0.2);

    // 標準偏差が近いことを確認
    EXPECT_NEAR(actual_std, stddev, 0.2);
}

TEST(RandomTest, NormalDistributionF64) {
    const int N = 100000;
    const double mean = 5.0;
    const double stddev = 0.5;
    std::vector<double> data(N);

    // f64 で生成
    ZenuStatus st = zenu_compute_normal_distribution_cpu(
        data.data(),
        N,
        static_cast<float>(mean),
        static_cast<float>(stddev),
        f64
    );
    ASSERT_EQ(st, Success);

    // 生成されたデータの平均と標準偏差を確認
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
    double actual_std = std::sqrt(sq_sum / (N-1));

    EXPECT_NEAR(actual_mean, mean, 0.05);
    EXPECT_NEAR(actual_std, stddev, 0.05);
}

// ------------------------------
// テスト: 一様分布
// ------------------------------
TEST(RandomTest, UniformDistributionF32) {
    const int N = 100000;
    const float low = -1.0f;
    const float high = 3.0f;
    std::vector<float> data(N);

    ZenuStatus st = zenu_compute_uniform_distribution_cpu(
        data.data(),
        N,
        low,
        high,
        f32
    );
    ASSERT_EQ(st, Success);

    // min, max を確認 (範囲外があれば失敗)
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

    // 平均を目安でチェック (一様分布 [low,high] => 理論平均は (low+high)/2 )
    double actual_mean = sum / N;
    double expected_mean = 0.5 * (low + high);
    EXPECT_NEAR(actual_mean, expected_mean, 0.1 * (high - low)); 
}

TEST(RandomTest, UniformDistributionF64) {
    const int N = 100000;
    const double low = 100.0;
    const double high = 200.0;
    std::vector<double> data(N);

    ZenuStatus st = zenu_compute_uniform_distribution_cpu(
        data.data(),
        N,
        static_cast<float>(low),
        static_cast<float>(high),
        f64
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

// main 関数 (GoogleTest が用意しているマクロ)
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

