#include "zenu_compute_conv.h"
#include "zenu_compute_type.h"
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <omp.h>

ZenuStatus zenu_compute_conv_forward_bias_cpu(
    size_t* input_shape,
    size_t* bias_shape,
    size_t conv_dim,
    ZenuDataType data_type,
    const void* input,
    const void* bias,
    void* output)
{
    if (!input_shape || !bias_shape || !input || !bias || !output) {
        return InvalidArgument;
    }
    if (data_type != ZenuDataType::f32 && data_type != ZenuDataType::f64) {
        return InvalidArgument;
    }

    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H, W;
    if (conv_dim == 1) {
        H = 1;
        W = input_shape[2];
    }
    else if (conv_dim == 2) {
        H = input_shape[2];
        W = input_shape[3];
    }
    else {
        return InvalidArgument;
    }
    size_t total = N * C * H * W;

    if (data_type == ZenuDataType::f32) {
        const float* in = static_cast<const float*>(input);
        const float* b = static_cast<const float*>(bias);
        float* out = static_cast<float*>(output);
        #pragma omp parallel for
        for (size_t idx = 0; idx < total; ++idx) {
            size_t c = (idx / (H * W)) % C;
            out[idx] = in[idx] + b[c];
        }
    }
    else {
        const double* in = static_cast<const double*>(input);
        const double* b = static_cast<const double*>(bias);
        double* out = static_cast<double*>(output);
        #pragma omp parallel for
        for (size_t idx = 0; idx < total; ++idx) {
            size_t c = (idx / (H * W)) % C;
            out[idx] = in[idx] + b[c];
        }
    }
    return Success;
}

ZenuStatus zenu_compute_conv_bkwd_bias_cpu(
    size_t* input_shape,
    size_t* bias_shape,
    size_t conv_dim,
    ZenuDataType data_type,
    const void* d_output,
    void* d_bias)
{
    if (!input_shape || !bias_shape || !d_output || !d_bias) {
        return InvalidArgument;
    }
    if (data_type != ZenuDataType::f32 && data_type != ZenuDataType::f64) {
        return InvalidArgument;
    }

    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H, W;
    if (conv_dim == 1) {
        H = 1;
        W = input_shape[2];
    }
    else if (conv_dim == 2) {
        H = input_shape[2];
        W = input_shape[3];
    }
    else {
        return InvalidArgument;
    }
    size_t spatial = H * W;

    if (data_type == ZenuDataType::f32) {
        const float* dout = static_cast<const float*>(d_output);
        float* dbias = static_cast<float*>(d_bias);
        #pragma omp parallel for
        for (size_t c = 0; c < C; c++) {
            float sum = 0.0f;
            for (size_t n = 0; n < N; n++) {
                for (size_t i = 0; i < spatial; i++) {
                    size_t index = n * (C * spatial) + c * spatial + i;
                    sum += dout[index];
                }
            }
            dbias[c] = sum;
        }
    }
    else {
        const double* dout = static_cast<const double*>(d_output);
        double* dbias = static_cast<double*>(d_bias);
        #pragma omp parallel for
        for (size_t c = 0; c < C; c++) {
            double sum = 0.0;
            for (size_t n = 0; n < N; n++) {
                for (size_t i = 0; i < spatial; i++) {
                    size_t index = n * (C * spatial) + c * spatial + i;
                    sum += dout[index];
                }
            }
            dbias[c] = sum;
        }
    }
    return Success;
}

