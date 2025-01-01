#include "zenu_compute_random.h"
#include <random>

// CPU normal
ZenuStatus zenu_compute_normal_distribution_cpu(
    void* dst,
    int   num_elements,
    float mean,
    float stddev,
    ZenuDataType data_type,
    unsigned long long seed
)
{
    if(!dst) return InvalidArgument;
    if(num_elements<=0) return InvalidArgument;
    if(stddev<0.f) return InvalidArgument;

    // 例: 64bitエンジン
    std::mt19937_64 rng(seed);

    if(data_type==f32){
        float* outF = (float*)dst;
        std::normal_distribution<float> dist(mean, stddev);
        for(int i=0; i<num_elements; i++){
            outF[i] = dist(rng);
        }
        return Success;
    }
    else if(data_type==f64){
        double* outD = (double*)dst;
        double dmean  = (double)mean;
        double dstd   = (double)stddev;
        std::normal_distribution<double> dist(dmean, dstd);
        for(int i=0; i<num_elements; i++){
            outD[i] = dist(rng);
        }
        return Success;
    }
    else{
        return InvalidArgument;
    }
}

// CPU uniform
ZenuStatus zenu_compute_uniform_distribution_cpu(
    void* dst,
    int   num_elements,
    float low,
    float high,
    ZenuDataType data_type,
    unsigned long long seed
)
{
    if(!dst) return InvalidArgument;
    if(num_elements<=0) return InvalidArgument;
    if(low>high) return InvalidArgument;

    std::mt19937_64 rng(seed);

    if(data_type==f32){
        float* outF = (float*)dst;
        std::uniform_real_distribution<float> dist(low, high);
        for(int i=0; i<num_elements; i++){
            outF[i] = dist(rng);
        }
        return Success;
    }
    else if(data_type==f64){
        double* outD = (double*)dst;
        double dlow  = (double)low;
        double dhigh = (double)high;
        std::uniform_real_distribution<double> dist(dlow, dhigh);
        for(int i=0; i<num_elements; i++){
            outD[i] = dist(rng);
        }
        return Success;
    }
    else{
        return InvalidArgument;
    }
}

