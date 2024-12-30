#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void conv2d_bias_bkwd_float(
    const float* dOut,  // device ptr
    float* dbias,       // device ptr
    int N, int C, int H, int W
);

void conv2d_bias_bkwd_double(
    const double* dOut,  // device ptr
    double* dbias,       // device ptr
    int N, int C, int H, int W
);

#ifdef __cplusplus
}
#endif
