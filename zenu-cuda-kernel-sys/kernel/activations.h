#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void relu_float(float *input, float *output, float alpha, int size, int input_stride, int output_stride);
void relu_double(double *input, double *output, double alpha, int size, int input_stride, int output_stride);

void relu_backward_mask_float(float *input, float *mask, float alpha, int size, int input_stride, int mask_stride);
void relu_backward_mask_double(double *input, double *mask, double alpha, int size, int input_stride, int mask_stride);

#ifdef __cplusplus
}
#endif
