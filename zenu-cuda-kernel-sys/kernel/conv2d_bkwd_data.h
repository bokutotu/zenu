#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void conv2d_bias_bkwd_float(
    const float* dOut, float* dbias, int N, int C, int H, int W
);

void conv2d_bias_bkwd_double(
    const double* dOut, double* dbias, int N, int C, int H, int W
);

#ifdef __cplusplus
}
#endif
