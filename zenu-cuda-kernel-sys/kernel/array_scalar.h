#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void array_scalar_add_float(float *a, int size, int stride_a, float scalar, float *out, int stride_out);
void array_scalar_sub_float(float *a, int size, int stride_a, float scalar, float *out, int stride_out);
void array_scalar_mul_float(float *a, int size, int stride_a, float scalar, float *out, int stride_out);
void array_scalar_div_float(float *a, int size, int stride_a, float scalar, float *out, int stride_out);
void array_scalar_add_double(double *a, int size, int stride_a, double scalar, double *out, int stride_out);
void array_scalar_sub_double(double *a, int size, int stride_a, double scalar, double *out, int stride_out);
void array_scalar_mul_double(double *a, int size, int stride_a, double scalar, double *out, int stride_out);
void array_scalar_div_double(double *a, int size, int stride_a, double scalar, double *out, int stride_out);
void array_scalar_add_assign_float(float *a, int size, int stride, float scalar);
void array_scalar_sub_assign_float(float *a, int size, int stride, float scalar);
void array_scalar_mul_assign_float(float *a, int size, int stride, float scalar);
void array_scalar_div_assign_float(float *a, int size, int stride, float scalar);
void array_scalar_add_assign_double(double *a, int size, int stride, double scalar);
void array_scalar_sub_assign_double(double *a, int size, int stride, double scalar);
void array_scalar_mul_assign_double(double *a, int size, int stride, double scalar);
void array_scalar_div_assign_double(double *a, int size, int stride, double scalar);

void array_sin_float(float *a, int size, int stride_in, float *out, int stride_out);
void array_cos_float(float *a, int size, int stride_in, float *out, int stride_out);
void array_tan_float(float *a, int size, int stride_in, float *out, int stride_out);
void array_asin_float(float *a, int size, int stride_in, float *out, int stride_out);
void array_acos_float(float *a, int size, int stride_in, float *out, int stride_out);
void array_atan_float(float *a, int size, int stride_in, float *out, int stride_out);
void array_sinh_float(float *a, int size, int stride_in, float *out, int stride_out);
void array_cosh_float(float *a, int size, int stride_in, float *out, int stride_out);
void array_tanh_float(float *a, int size, int stride_in, float *out, int stride_out);
void array_abs_float(float *a, int size, int stride_in, float *out, int stride_out);
void array_sqrt_float(float *a, int size, int stride_in, float *out, int stride_out);
void array_exp_float(float *a, int size, int stride_in, float *out, int stride_out);

void array_sin_double(double *a, int size, int stride_in, double *out, int stride_out);
void array_cos_double(double *a, int size, int stride_in, double *out, int stride_out);
void array_tan_double(double *a, int size, int stride_in, double *out, int stride_out);
void array_asin_double(double *a, int size, int stride_in, double *out, int stride_out);
void array_acos_double(double *a, int size, int stride_in, double *out, int stride_out);
void array_atan_double(double *a, int size, int stride_in, double *out, int stride_out);
void array_sinh_double(double *a, int size, int stride_in, double *out, int stride_out);
void array_cosh_double(double *a, int size, int stride_in, double *out, int stride_out);
void array_tanh_double(double *a, int size, int stride_in, double *out, int stride_out);
void array_abs_double(double *a, int size, int stride_in, double *out, int stride_out);
void array_sqrt_double(double *a, int size, int stride_in, double *out, int stride_out);
void array_exp_double(double *a, int size, int stride_in, double *out, int stride_out);

void array_sin_assign_float(float *a, int size, int stride);
void array_cos_assign_float(float *a, int size, int stride);
void array_tan_assign_float(float *a, int size, int stride);
void array_asin_assign_float(float *a, int size, int stride);
void array_acos_assign_float(float *a, int size, int stride);
void array_atan_assign_float(float *a, int size, int stride);
void array_sinh_assign_float(float *a, int size, int stride);
void array_cosh_assign_float(float *a, int size, int stride);
void array_tanh_assign_float(float *a, int size, int stride);
void array_abs_assign_float(float *a, int size, int stride);
void array_sqrt_assign_float(float *a, int size, int stride);
void array_exp_assign_float(float *a, int size, int stride);

void array_sin_assign_double(double *a, int size, int stride);
void array_cos_assign_double(double *a, int size, int stride);
void array_tan_assign_double(double *a, int size, int stride);
void array_asin_assign_double(double *a, int size, int stride);
void array_acos_assign_double(double *a, int size, int stride);
void array_atan_assign_double(double *a, int size, int stride);
void array_sinh_assign_double(double *a, int size, int stride);
void array_cosh_assign_double(double *a, int size, int stride);
void array_tanh_assign_double(double *a, int size, int stride);
void array_abs_assign_double(double *a, int size, int stride);
void array_sqrt_assign_double(double *a, int size, int stride);
void array_exp_assign_double(double *a, int size, int stride);

void array_clip_float(float* input, float *output, int size, int stride_in, int stride_out, float min, float max);
void array_clip_double(double* input, double *output, int size, int stride_in, int stride_out, double min, double max);
void array_clip_assign_float(float* input, int size, int stride, float min, float max);
void array_clip_assign_double(double* input, int size, int stride, double min, double max);

#ifdef __cplusplus
}
#endif
