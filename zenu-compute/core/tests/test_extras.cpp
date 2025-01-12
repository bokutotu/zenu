#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "zenu_compute.h"
#include "array_comp.h"

//--------------------------------------------------------------------------
// ヘルパー関数: CPU上でテスト入力用の配列を作成 (float / double)
//--------------------------------------------------------------------------
std::vector<float> make_test_array_f32() {
    // 例として -1, 0, 1, 2, 3, -2 のような入力
    return std::vector<float>{-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, -2.0f};
}

std::vector<double> make_test_array_f64() {
    return std::vector<double>{-1.0, 0.0, 1.0, 2.0, 3.0, -2.0};
}

//--------------------------------------------------------------------------
// ヘルパー関数: デバイス (nvidia) にホスト配列をコピーして、
//               テスト呼び出し後、結果をホストに持ってきて比較する
//--------------------------------------------------------------------------
bool compare_device_result_f32(const float* device_ptr, const float* expected, size_t n, float tol = 1e-5f)
{
    // GPU->CPU にコピー
    std::vector<float> host_result(n, 0.0f);
    ZenuStatus st = zenu_compute_nvidia_to_cpu(host_result.data(), (void*)device_ptr, sizeof(float)*n);
    if (st != ZenuStatus::Success) {
        std::cerr << "Error: zenu_compute_nvidia_to_cpu() failed\n";
        return false;
    }
    // array_comp.h を用いて比較
    return zenu_compare_array_f32(host_result.data(), expected, n, tol);
}

bool compare_device_result_f64(const double* device_ptr, const double* expected, size_t n, double tol = 1e-9)
{
    std::vector<double> host_result(n, 0.0);
    ZenuStatus st = zenu_compute_nvidia_to_cpu(host_result.data(), (void*)device_ptr, sizeof(double)*n);
    if (st != ZenuStatus::Success) {
        std::cerr << "Error: zenu_compute_nvidia_to_cpu() failed\n";
        return false;
    }
    return zenu_compare_array_f64(host_result.data(), expected, n, tol);
}

//--------------------------------------------------------------------------
// exp, ln, abs, clip, pow のテスト用に、
// 期待値(ground truth)を計算する小さなユーティリティ
//--------------------------------------------------------------------------
template<typename T>
static inline T my_exp(T x) { return std::exp(x); }

template<typename T>
static inline T my_ln(T x) { return std::log(x); }

template<typename T>
static inline T my_abs(T x) { return std::fabs(x); }

template<typename T>
static inline T my_clip(T x, T minv, T maxv) {
    if (x < minv) return minv;
    if (x > maxv) return maxv;
    return x;
}

template<typename T>
static inline T my_pow(T x, T expval) { return std::pow(x, expval); }

//--------------------------------------------------------------------------
// テスト開始
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, ExpMatCPU_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();

    // 出力バッファを確保
    std::vector<float> output(n, 0.0f);

    // 期待結果を先に計算
    std::vector<float> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_exp(input[i]);
    }

    // 実際に呼び出し
    ZenuStatus st = zenu_compute_exp_mat_cpu(
        output.data(),
        input.data(),
        /*stride_dst=*/1,
        /*stride_src=*/1,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 比較
    EXPECT_TRUE(zenu_compare_array_f32(output.data(), expected.data(), n));
}

TEST(TestZenuComputeExtras, ExpMatCPU_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();
    std::vector<double> output(n, 0.0);
    std::vector<double> expected(n);

    for(size_t i=0; i<n; i++){
        expected[i] = my_exp(input[i]);
    }

    ZenuStatus st = zenu_compute_exp_mat_cpu(
        output.data(),
        input.data(),
        1,
        1,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    EXPECT_TRUE(zenu_compare_array_f64(output.data(), expected.data(), n));
}

//--------------------------------------------------------------------------
// exp nvidia out-of-place
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, ExpMatNvidia_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();

    // GPU バッファ確保 (in, out)
    void* d_in = nullptr;
    void* d_out = nullptr;
    ZenuStatus st;
    st = zenu_compute_malloc_nvidia(&d_in,  sizeof(float)*n); ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_malloc_nvidia(&d_out, sizeof(float)*n); ASSERT_EQ(st, ZenuStatus::Success);

    // CPU->GPU
    st = zenu_compute_cpu_to_nvidia(d_in, input.data(), sizeof(float)*n);  
    ASSERT_EQ(st, ZenuStatus::Success);

    // 演算実行
    st = zenu_compute_exp_mat_nvidia(
        d_out,
        d_in,
        /*stride_dst=*/1,
        /*stride_src=*/1,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 期待値計算
    std::vector<float> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_exp(input[i]);
    }

    // 比較 (GPU -> CPUして array_comp)
    EXPECT_TRUE(compare_device_result_f32((float*)d_out, expected.data(), n));

    // 後始末
    zenu_compute_free_nvidia(d_in);
    zenu_compute_free_nvidia(d_out);
}

TEST(TestZenuComputeExtras, ExpMatNvidia_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();

    void* d_in = nullptr;
    void* d_out = nullptr;
    ZenuStatus st;
    st = zenu_compute_malloc_nvidia(&d_in,  sizeof(double)*n); ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_malloc_nvidia(&d_out, sizeof(double)*n); ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_in, input.data(), sizeof(double)*n);  
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_exp_mat_nvidia(
        d_out,
        d_in,
        1,
        1,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<double> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_exp(input[i]);
    }

    EXPECT_TRUE(compare_device_result_f64((double*)d_out, expected.data(), n));

    zenu_compute_free_nvidia(d_in);
    zenu_compute_free_nvidia(d_out);
}

//--------------------------------------------------------------------------
// exp assign cpu
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, ExpMatAssignCPU_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();

    // in-place なので dst として同じバッファを使う
    std::vector<float> buf = input;

    // 呼び出し
    ZenuStatus st = zenu_compute_exp_mat_assign_cpu(
        buf.data(),
        /*stride_dst=*/1,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 期待値
    std::vector<float> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_exp(input[i]);
    }

    EXPECT_TRUE(zenu_compare_array_f32(buf.data(), expected.data(), n));
}

TEST(TestZenuComputeExtras, ExpMatAssignCPU_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();

    std::vector<double> buf = input;

    ZenuStatus st = zenu_compute_exp_mat_assign_cpu(
        buf.data(),
        1,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<double> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_exp(input[i]);
    }

    EXPECT_TRUE(zenu_compare_array_f64(buf.data(), expected.data(), n));
}

//--------------------------------------------------------------------------
// exp assign nvidia
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, ExpMatAssignNvidia_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();

    void* d_buf = nullptr;
    ZenuStatus st = zenu_compute_malloc_nvidia(&d_buf, sizeof(float)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    // 入力をGPUへコピー
    st = zenu_compute_cpu_to_nvidia(d_buf, input.data(), sizeof(float)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    // in-place
    st = zenu_compute_exp_mat_assign_nvidia(
        d_buf,
        /*stride_dst=*/1,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 期待値
    std::vector<float> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_exp(input[i]);
    }

    EXPECT_TRUE(compare_device_result_f32((float*)d_buf, expected.data(), n));

    zenu_compute_free_nvidia(d_buf);
}

TEST(TestZenuComputeExtras, ExpMatAssignNvidia_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();

    void* d_buf = nullptr;
    ZenuStatus st = zenu_compute_malloc_nvidia(&d_buf, sizeof(double)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_buf, input.data(), sizeof(double)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_exp_mat_assign_nvidia(
        d_buf,
        1,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<double> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_exp(input[i]);
    }

    EXPECT_TRUE(compare_device_result_f64((double*)d_buf, expected.data(), n));

    zenu_compute_free_nvidia(d_buf);
}

//--------------------------------------------------------------------------
// ln cpu
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, LnMatCPU_F32)
{
    // ln は入力に負数が含まれると NaN になるので、正の値でテスト
    std::vector<float> input{0.1f, 1.0f, 2.0f, 10.0f};
    size_t n = input.size();
    std::vector<float> output(n, 0.0f);

    std::vector<float> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_ln(input[i]);
    }

    ZenuStatus st = zenu_compute_ln_mat_cpu(
        output.data(),
        input.data(),
        1,
        1,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 近似チェック
    EXPECT_TRUE(zenu_compare_array_f32(output.data(), expected.data(), n, 1e-5f));
}

TEST(TestZenuComputeExtras, LnMatCPU_F64)
{
    std::vector<double> input{0.1, 1.0, 2.0, 10.0};
    size_t n = input.size();
    std::vector<double> output(n, 0.0);

    std::vector<double> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_ln(input[i]);
    }

    ZenuStatus st = zenu_compute_ln_mat_cpu(
        output.data(),
        input.data(),
        1,
        1,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    EXPECT_TRUE(zenu_compare_array_f64(output.data(), expected.data(), n, 1e-9));
}

//--------------------------------------------------------------------------
// ln nvidia
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, LnMatNvidia_F32)
{
    std::vector<float> input{0.1f, 1.0f, 2.0f, 10.0f};
    size_t n = input.size();

    void* d_in = nullptr;
    void* d_out = nullptr;
    ZenuStatus st;
    st = zenu_compute_malloc_nvidia(&d_in,  sizeof(float)*n); ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_malloc_nvidia(&d_out, sizeof(float)*n); ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_in, input.data(), sizeof(float)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_ln_mat_nvidia(
        d_out,
        d_in,
        1,
        1,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 期待値
    std::vector<float> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_ln(input[i]);
    }

    EXPECT_TRUE(compare_device_result_f32((float*)d_out, expected.data(), n, 1e-5f));

    zenu_compute_free_nvidia(d_in);
    zenu_compute_free_nvidia(d_out);
}

TEST(TestZenuComputeExtras, LnMatNvidia_F64)
{
    std::vector<double> input{0.1, 1.0, 2.0, 10.0};
    size_t n = input.size();

    void* d_in = nullptr;
    void* d_out = nullptr;
    ZenuStatus st;
    st = zenu_compute_malloc_nvidia(&d_in,  sizeof(double)*n); ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_malloc_nvidia(&d_out, sizeof(double)*n); ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_in, input.data(), sizeof(double)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_ln_mat_nvidia(
        d_out,
        d_in,
        1,
        1,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<double> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_ln(input[i]);
    }

    EXPECT_TRUE(compare_device_result_f64((double*)d_out, expected.data(), n, 1e-9));

    zenu_compute_free_nvidia(d_in);
    zenu_compute_free_nvidia(d_out);
}

//--------------------------------------------------------------------------
// ln assign cpu
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, LnMatAssignCPU_F32)
{
    std::vector<float> buf{0.1f, 1.0f, 2.0f, 10.0f};
    size_t n = buf.size();

    ZenuStatus st = zenu_compute_ln_mat_assign_cpu(
        buf.data(),
        1,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<float> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_ln((float[]){0.1f, 1.0f, 2.0f, 10.0f}[i]);
    }

    EXPECT_TRUE(zenu_compare_array_f32(buf.data(), expected.data(), n, 1e-5f));
}

TEST(TestZenuComputeExtras, LnMatAssignCPU_F64)
{
    std::vector<double> buf{0.1, 1.0, 2.0, 10.0};
    size_t n = buf.size();

    ZenuStatus st = zenu_compute_ln_mat_assign_cpu(
        buf.data(),
        1,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<double> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_ln((double[]){0.1, 1.0, 2.0, 10.0}[i]);
    }

    EXPECT_TRUE(zenu_compare_array_f64(buf.data(), expected.data(), n, 1e-9));
}

//--------------------------------------------------------------------------
// ln assign nvidia
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, LnMatAssignNvidia_F32)
{
    std::vector<float> input{0.1f, 1.0f, 2.0f, 10.0f};
    size_t n = input.size();

    void* d_buf = nullptr;
    ZenuStatus st = zenu_compute_malloc_nvidia(&d_buf, sizeof(float)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_buf, input.data(), sizeof(float)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_ln_mat_assign_nvidia(
        d_buf,
        1,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 期待値
    std::vector<float> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_ln(input[i]);
    }

    EXPECT_TRUE(compare_device_result_f32((float*)d_buf, expected.data(), n, 1e-5f));

    zenu_compute_free_nvidia(d_buf);
}

TEST(TestZenuComputeExtras, LnMatAssignNvidia_F64)
{
    std::vector<double> input{0.1, 1.0, 2.0, 10.0};
    size_t n = input.size();

    void* d_buf = nullptr;
    ZenuStatus st = zenu_compute_malloc_nvidia(&d_buf, sizeof(double)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_buf, input.data(), sizeof(double)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_ln_mat_assign_nvidia(
        d_buf,
        1,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<double> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_ln(input[i]);
    }

    EXPECT_TRUE(compare_device_result_f64((double*)d_buf, expected.data(), n, 1e-9));

    zenu_compute_free_nvidia(d_buf);
}

//--------------------------------------------------------------------------
// abs cpu
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, AbsMatCPU_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();

    std::vector<float> output(n, 0.0f);
    ZenuStatus st = zenu_compute_abs_mat_cpu(
        output.data(),
        input.data(),
        1,
        1,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 期待値
    std::vector<float> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_abs(input[i]);
    }

    EXPECT_TRUE(zenu_compare_array_f32(output.data(), expected.data(), n));
}

TEST(TestZenuComputeExtras, AbsMatCPU_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();

    std::vector<double> output(n, 0.0);
    ZenuStatus st = zenu_compute_abs_mat_cpu(
        output.data(),
        input.data(),
        1,
        1,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<double> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_abs(input[i]);
    }

    EXPECT_TRUE(zenu_compare_array_f64(output.data(), expected.data(), n));
}

//--------------------------------------------------------------------------
// abs nvidia
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, AbsMatNvidia_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();

    void* d_in = nullptr;
    void* d_out = nullptr;
    ZenuStatus st;
    st = zenu_compute_malloc_nvidia(&d_in,  sizeof(float)*n); ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_malloc_nvidia(&d_out, sizeof(float)*n); ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_in, input.data(), sizeof(float)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_abs_mat_nvidia(
        d_out,
        d_in,
        1,
        1,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<float> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_abs(input[i]);
    }

    EXPECT_TRUE(compare_device_result_f32((float*)d_out, expected.data(), n));

    zenu_compute_free_nvidia(d_in);
    zenu_compute_free_nvidia(d_out);
}

TEST(TestZenuComputeExtras, AbsMatNvidia_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();

    void* d_in = nullptr;
    void* d_out = nullptr;
    ZenuStatus st;
    st = zenu_compute_malloc_nvidia(&d_in,  sizeof(double)*n); ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_malloc_nvidia(&d_out, sizeof(double)*n); ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_in, input.data(), sizeof(double)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_abs_mat_nvidia(
        d_out,
        d_in,
        1,
        1,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<double> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_abs(input[i]);
    }

    EXPECT_TRUE(compare_device_result_f64((double*)d_out, expected.data(), n));

    zenu_compute_free_nvidia(d_in);
    zenu_compute_free_nvidia(d_out);
}

//--------------------------------------------------------------------------
// abs assign cpu
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, AbsMatAssignCPU_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();

    std::vector<float> buf = input; // in-place
    ZenuStatus st = zenu_compute_abs_mat_assign_cpu(
        buf.data(),
        1,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 期待値
    for(size_t i=0; i<n; i++){
        input[i] = my_abs(input[i]);
    }
    EXPECT_TRUE(zenu_compare_array_f32(buf.data(), input.data(), n));
}

TEST(TestZenuComputeExtras, AbsMatAssignCPU_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();

    std::vector<double> buf = input;
    ZenuStatus st = zenu_compute_abs_mat_assign_cpu(
        buf.data(),
        1,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    for(size_t i=0; i<n; i++){
        input[i] = my_abs(input[i]);
    }
    EXPECT_TRUE(zenu_compare_array_f64(buf.data(), input.data(), n));
}

//--------------------------------------------------------------------------
// abs assign nvidia
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, AbsMatAssignNvidia_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();

    void* d_buf = nullptr;
    ZenuStatus st = zenu_compute_malloc_nvidia(&d_buf, sizeof(float)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_buf, input.data(), sizeof(float)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_abs_mat_assign_nvidia(
        d_buf,
        1,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 期待値
    for(size_t i=0; i<n; i++){
        input[i] = my_abs(input[i]);
    }

    EXPECT_TRUE(compare_device_result_f32((float*)d_buf, input.data(), n));

    zenu_compute_free_nvidia(d_buf);
}

TEST(TestZenuComputeExtras, AbsMatAssignNvidia_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();

    void* d_buf = nullptr;
    ZenuStatus st = zenu_compute_malloc_nvidia(&d_buf, sizeof(double)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_buf, input.data(), sizeof(double)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_abs_mat_assign_nvidia(
        d_buf,
        1,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    for(size_t i=0; i<n; i++){
        input[i] = my_abs(input[i]);
    }

    EXPECT_TRUE(compare_device_result_f64((double*)d_buf, input.data(), n));

    zenu_compute_free_nvidia(d_buf);
}

//--------------------------------------------------------------------------
// clip cpu
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, ClipMatCPU_F32)
{
    auto input = make_test_array_f32(); // [-1,0,1,2,3,-2]
    size_t n = input.size();

    float min_val = -0.5f;
    float max_val = 2.5f;

    std::vector<float> output(n, 0.0f);

    ZenuStatus st = zenu_compute_clip_mat_cpu(
        output.data(),
        input.data(),
        1,
        1,
        n,
        ZenuDataType::f32,
        (double)min_val,
        (double)max_val
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 期待値
    std::vector<float> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_clip(input[i], min_val, max_val);
    }

    EXPECT_TRUE(zenu_compare_array_f32(output.data(), expected.data(), n));
}

TEST(TestZenuComputeExtras, ClipMatCPU_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();

    double min_val = -0.5;
    double max_val = 2.5;

    std::vector<double> output(n, 0.0);

    ZenuStatus st = zenu_compute_clip_mat_cpu(
        output.data(),
        input.data(),
        1,
        1,
        n,
        ZenuDataType::f64,
        min_val,
        max_val
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<double> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_clip(input[i], min_val, max_val);
    }

    EXPECT_TRUE(zenu_compare_array_f64(output.data(), expected.data(), n));
}

//--------------------------------------------------------------------------
// clip nvidia
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, ClipMatNvidia_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();

    float min_val = -0.5f;
    float max_val = 2.5f;

    void* d_in = nullptr;
    void* d_out = nullptr;
    ZenuStatus st;
    st = zenu_compute_malloc_nvidia(&d_in,  sizeof(float)*n); ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_malloc_nvidia(&d_out, sizeof(float)*n); ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_in, input.data(), sizeof(float)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_clip_mat_nvidia(
        d_out,
        d_in,
        1,
        1,
        n,
        ZenuDataType::f32,
        (double)min_val,
        (double)max_val
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 期待値
    std::vector<float> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_clip(input[i], min_val, max_val);
    }

    EXPECT_TRUE(compare_device_result_f32((float*)d_out, expected.data(), n));

    zenu_compute_free_nvidia(d_in);
    zenu_compute_free_nvidia(d_out);
}

TEST(TestZenuComputeExtras, ClipMatNvidia_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();

    double min_val = -0.5;
    double max_val = 2.5;

    void* d_in = nullptr;
    void* d_out = nullptr;
    ZenuStatus st;
    st = zenu_compute_malloc_nvidia(&d_in,  sizeof(double)*n); ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_malloc_nvidia(&d_out, sizeof(double)*n); ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_in, input.data(), sizeof(double)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_clip_mat_nvidia(
        d_out,
        d_in,
        1,
        1,
        n,
        ZenuDataType::f64,
        min_val,
        max_val
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<double> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_clip(input[i], min_val, max_val);
    }

    EXPECT_TRUE(compare_device_result_f64((double*)d_out, expected.data(), n));

    zenu_compute_free_nvidia(d_in);
    zenu_compute_free_nvidia(d_out);
}

//--------------------------------------------------------------------------
// clip assign cpu
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, ClipMatAssignCPU_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();

    float min_val = -0.5f;
    float max_val = 2.5f;

    std::vector<float> buf = input; // in-place
    ZenuStatus st = zenu_compute_clip_mat_assign_cpu(
        buf.data(),
        1,
        n,
        ZenuDataType::f32,
        (double)min_val,
        (double)max_val
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    for(size_t i=0; i<n; i++){
        input[i] = my_clip(input[i], min_val, max_val);
    }
    EXPECT_TRUE(zenu_compare_array_f32(buf.data(), input.data(), n));
}

TEST(TestZenuComputeExtras, ClipMatAssignCPU_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();

    double min_val = -0.5;
    double max_val = 2.5;

    std::vector<double> buf = input;
    ZenuStatus st = zenu_compute_clip_mat_assign_cpu(
        buf.data(),
        1,
        n,
        ZenuDataType::f64,
        min_val,
        max_val
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    for(size_t i=0; i<n; i++){
        input[i] = my_clip(input[i], min_val, max_val);
    }
    EXPECT_TRUE(zenu_compare_array_f64(buf.data(), input.data(), n));
}

//--------------------------------------------------------------------------
// clip assign nvidia
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, ClipMatAssignNvidia_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();

    float min_val = -0.5f;
    float max_val = 2.5f;

    void* d_buf = nullptr;
    ZenuStatus st = zenu_compute_malloc_nvidia(&d_buf, sizeof(float)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_buf, input.data(), sizeof(float)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_clip_mat_assign_nvidia(
        d_buf,
        1,
        n,
        ZenuDataType::f32,
        (double)min_val,
        (double)max_val
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    for(size_t i=0; i<n; i++){
        input[i] = my_clip(input[i], min_val, max_val);
    }

    EXPECT_TRUE(compare_device_result_f32((float*)d_buf, input.data(), n));

    zenu_compute_free_nvidia(d_buf);
}

TEST(TestZenuComputeExtras, ClipMatAssignNvidia_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();

    double min_val = -0.5;
    double max_val = 2.5;

    void* d_buf = nullptr;
    ZenuStatus st = zenu_compute_malloc_nvidia(&d_buf, sizeof(double)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_buf, input.data(), sizeof(double)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_clip_mat_assign_nvidia(
        d_buf,
        1,
        n,
        ZenuDataType::f64,
        min_val,
        max_val
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    for(size_t i=0; i<n; i++){
        input[i] = my_clip(input[i], min_val, max_val);
    }

    EXPECT_TRUE(compare_device_result_f64((double*)d_buf, input.data(), n));

    zenu_compute_free_nvidia(d_buf);
}

//--------------------------------------------------------------------------
// pow cpu
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, PowMatCPU_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();
    float exponent = 2.0f;

    std::vector<float> output(n, 0.0f);
    ZenuStatus st = zenu_compute_pow_mat_cpu(
        output.data(),
        input.data(),
        1,
        1,
        (void*)&exponent,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 期待値
    std::vector<float> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_pow(input[i], exponent);
    }

    EXPECT_TRUE(zenu_compare_array_f32(output.data(), expected.data(), n));
}

TEST(TestZenuComputeExtras, PowMatCPU_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();
    double exponent = 2.0;

    std::vector<double> output(n, 0.0);
    ZenuStatus st = zenu_compute_pow_mat_cpu(
        output.data(),
        input.data(),
        1,
        1,
        (void*)&exponent,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    std::vector<double> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_pow(input[i], exponent);
    }

    EXPECT_TRUE(zenu_compare_array_f64(output.data(), expected.data(), n));
}

//--------------------------------------------------------------------------
// pow nvidia
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, PowMatNvidia_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();
    float exponent = 2.0f;

    void* d_in = nullptr;
    void* d_out = nullptr;
    void* device_exponent = nullptr;
    ZenuStatus st;
    st = zenu_compute_malloc_nvidia(&d_in,  sizeof(float)*n); ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_malloc_nvidia(&d_out, sizeof(float)*n); ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_malloc_nvidia(&device_exponent, sizeof(float));
    ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_cpu_to_nvidia(device_exponent, &exponent, sizeof(float));

    st = zenu_compute_cpu_to_nvidia(d_in, input.data(), sizeof(float)*n);
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_pow_mat_nvidia(
        d_out,
        d_in,
        1,
        1,
        (float*)device_exponent,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 期待値
    std::vector<float> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_pow(input[i], exponent);
    }

    EXPECT_TRUE(compare_device_result_f32((float*)d_out, expected.data(), n));

    zenu_compute_free_nvidia(d_in);
    zenu_compute_free_nvidia(d_out);
}

TEST(TestZenuComputeExtras, PowMatNvidia_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();
    double exponent = 2.0;

    void* d_in = nullptr;
    void* d_out = nullptr;
    void* device_exponent = nullptr;
    ZenuStatus st;
    st = zenu_compute_malloc_nvidia(&d_in,  sizeof(double)*n); ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_malloc_nvidia(&d_out, sizeof(double)*n); ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_malloc_nvidia(&device_exponent, sizeof(double)); ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_in, input.data(), sizeof(double)*n); ASSERT_EQ(st, ZenuStatus::Success);
    st = zenu_compute_cpu_to_nvidia(device_exponent, &exponent, sizeof(double)); ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_pow_mat_nvidia(
        d_out,
        d_in,
        1,
        1,
        (void*)device_exponent,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    // 期待値
    std::vector<double> expected(n);
    for(size_t i=0; i<n; i++){
        expected[i] = my_pow(input[i], exponent);
    }

    EXPECT_TRUE(compare_device_result_f64((double*)d_out, expected.data(), n));

    zenu_compute_free_nvidia(d_in);
    zenu_compute_free_nvidia(d_out);
}

//--------------------------------------------------------------------------
// pow assign cpu
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, PowMatAssignCPU_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();
    float exponent = 3.0f;

    std::vector<float> buf = input; // in-place
    ZenuStatus st = zenu_compute_pow_mat_assign_cpu(
        buf.data(),
        1,
        (void*)&exponent,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    for(size_t i=0; i<n; i++){
        input[i] = my_pow(input[i], exponent);
    }
    EXPECT_TRUE(zenu_compare_array_f32(buf.data(), input.data(), n));
}

TEST(TestZenuComputeExtras, PowMatAssignCPU_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();
    double exponent = 3.0;

    std::vector<double> buf = input;
    ZenuStatus st = zenu_compute_pow_mat_assign_cpu(
        buf.data(),
        1,
        (void*)&exponent,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    for(size_t i=0; i<n; i++){
        input[i] = my_pow(input[i], exponent);
    }
    EXPECT_TRUE(zenu_compare_array_f64(buf.data(), input.data(), n));
}

//--------------------------------------------------------------------------
// pow assign nvidia
//--------------------------------------------------------------------------

TEST(TestZenuComputeExtras, PowMatAssignNvidia_F32)
{
    auto input = make_test_array_f32();
    size_t n = input.size();
    float exponent = 3.0f;

    void* d_buf = nullptr;
    void* device_exponent = nullptr;
    ZenuStatus st = zenu_compute_malloc_nvidia(&d_buf, sizeof(float)*n);
    st = zenu_compute_malloc_nvidia(&device_exponent, sizeof(float));
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_buf, input.data(), sizeof(float)*n);
    st = zenu_compute_cpu_to_nvidia(device_exponent, &exponent, sizeof(float));
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_pow_mat_assign_nvidia(
        d_buf,
        1,
        (void*)device_exponent,
        n,
        ZenuDataType::f32
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    for(size_t i=0; i<n; i++){
        input[i] = my_pow(input[i], exponent);
    }

    EXPECT_TRUE(compare_device_result_f32((float*)d_buf, input.data(), n));

    zenu_compute_free_nvidia(d_buf);
}

TEST(TestZenuComputeExtras, PowMatAssignNvidia_F64)
{
    auto input = make_test_array_f64();
    size_t n = input.size();
    double exponent = 3.0;

    void* d_buf = nullptr;
    void* device_exponent = nullptr;
    ZenuStatus st = zenu_compute_malloc_nvidia(&d_buf, sizeof(double)*n);
    st = zenu_compute_malloc_nvidia(&device_exponent, sizeof(double));
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_cpu_to_nvidia(d_buf, input.data(), sizeof(double)*n);
    st = zenu_compute_cpu_to_nvidia(device_exponent, &exponent, sizeof(double));
    ASSERT_EQ(st, ZenuStatus::Success);

    st = zenu_compute_pow_mat_assign_nvidia(
        d_buf,
        1,
        (void*)device_exponent,
        n,
        ZenuDataType::f64
    );
    ASSERT_EQ(st, ZenuStatus::Success);

    for(size_t i=0; i<n; i++){
        input[i] = my_pow(input[i], exponent);
    }

    EXPECT_TRUE(compare_device_result_f64((double*)d_buf, input.data(), n));

    zenu_compute_free_nvidia(d_buf);
}

