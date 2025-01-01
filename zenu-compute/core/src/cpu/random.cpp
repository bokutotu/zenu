#include "zenu_compute.h"

#include <random>     // std::mt19937, std::uniform_real_distribution, std::normal_distribution
#include <cmath>      // for double/float

//=================================================================
// CPU上で一様分布乱数を生成 (スカラー, SIMDなし)
//=================================================================
extern "C"
ZenuStatus zenu_compute_uniform_distribution_cpu(void* dst,
                                                 int   num_elements,
                                                 float low,
                                                 float high,
                                                 ZenuDataType data_type)
{
    // --- バリデーション ---
    if (!dst)              return InvalidArgument;
    if (num_elements <= 0) return InvalidArgument;
    if (low >= high)       return InvalidArgument;

    // 乱数エンジン (例: Mersenne Twister)
    // シードは固定 or std::random_device{}() など
    std::mt19937 rng(1234);  
    // シードを変えたい場合、引数追加など適宜調整

    if (data_type == f32)
    {
        float* outF = static_cast<float*>(dst);
        std::uniform_real_distribution<float> dist(low, high);

        for (int i = 0; i < num_elements; i++) {
            outF[i] = dist(rng);
        }
        return Success;
    }
    else if (data_type == f64)
    {
        double* outD = static_cast<double*>(dst);
        double dlow  = static_cast<double>(low);
        double dhigh = static_cast<double>(high);
        std::uniform_real_distribution<double> dist(dlow, dhigh);

        for (int i = 0; i < num_elements; i++) {
            outD[i] = dist(rng);
        }
        return Success;
    }
    else
    {
        return InvalidArgument;
    }
}

//=================================================================
// CPU上で正規分布乱数を生成 (スカラー, SIMDなし)
//=================================================================
extern "C"
ZenuStatus zenu_compute_normal_distribution_cpu(void* dst,
                                                int   num_elements,
                                                float mean,
                                                float stddev,
                                                ZenuDataType data_type)
{
    // --- バリデーション ---
    if (!dst)                  return InvalidArgument;
    if (num_elements <= 0)     return InvalidArgument;
    if (stddev < 0.0f)         return InvalidArgument;

    // 乱数エンジン
    std::mt19937 rng(4321);  // シード固定の例

    if (data_type == f32)
    {
        float* outF = static_cast<float*>(dst);
        std::normal_distribution<float> dist(mean, stddev);

        for (int i = 0; i < num_elements; i++) {
            outF[i] = dist(rng);
        }
        return Success;
    }
    else if (data_type == f64)
    {
        double* outD = static_cast<double*>(dst);
        double dmean  = (double)mean;
        double dstd   = (double)stddev;
        std::normal_distribution<double> dist(dmean, dstd);

        for (int i = 0; i < num_elements; i++) {
            outD[i] = dist(rng);
        }
        return Success;
    }
    else
    {
        return InvalidArgument;
    }
}

