#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void array_array_add_float(float *a, int stride_a, float *b, int stride_b, float *c, int stride_c, int n);
void array_array_sub_float(float *a, int stride_a, float *b, int stride_b, float *c, int stride_c, int n);
void array_array_mul_float(float *a, int stride_a, float *b, int stride_b, float *c, int stride_c, int n);
void array_array_div_float(float *a, int stride_a, float *b, int stride_b, float *c, int stride_c, int n);

void array_array_add_double(double *a, int stride_a, double *b, int stride_b, double *c, int stride_c, int n);
void array_array_sub_double(double *a, int stride_a, double *b, int stride_b, double *c, int stride_c, int n);
void array_array_mul_double(double *a, int stride_a, double *b, int stride_b, double *c, int stride_c, int n);
void array_array_div_double(double *a, int stride_a, double *b, int stride_b, double *c, int stride_c, int n);

void array_array_add_assign_float(float *a, int stride_a, float *b, int stride_b, int n);
void array_array_sub_assign_float(float *a, int stride_a, float *b, int stride_b, int n);
void array_array_mul_assign_float(float *a, int stride_a, float *b, int stride_b, int n);
void array_array_div_assign_float(float *a, int stride_a, float *b, int stride_b, int n);

void array_array_add_assign_double(double *a, int stride_a, double *b, int stride_b, int n);
void array_array_sub_assign_double(double *a, int stride_a, double *b, int stride_b, int n);
void array_array_mul_assign_double(double *a, int stride_a, double *b, int stride_b, int n);
void array_array_div_assign_double(double *a, int stride_a, double *b, int stride_b, int n);

#ifdef __cplusplus
}
#endif
