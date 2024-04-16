#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void array_scalar_add_float(float *a, int size, int stride, float scalar, float *out);
void array_scalar_sub_float(float *a, int size, int stride, float scalar, float *out);
void array_scalar_mul_float(float *a, int size, int stride, float scalar, float *out);
void array_scalar_div_float(float *a, int size, int stride, float scalar, float *out);
void array_scalar_add_double(double *a, int size, int stride, double scalar, double *out);
void array_scalar_sub_double(double *a, int size, int stride, double scalar, double *out);
void array_scalar_mul_double(double *a, int size, int stride, double scalar, double *out);
void array_scalar_div_double(double *a, int size, int stride, double scalar, double *out);
void array_scalar_add_assign_float(float *a, int size, int stride, float scalar);
void array_scalar_sub_assign_float(float *a, int size, int stride, float scalar);
void array_scalar_mul_assign_float(float *a, int size, int stride, float scalar);
void array_scalar_div_assign_float(float *a, int size, int stride, float scalar);
void array_scalar_add_assign_double(double *a, int size, int stride, double scalar);
void array_scalar_sub_assign_double(double *a, int size, int stride, double scalar);
void array_scalar_mul_assign_double(double *a, int size, int stride, double scalar);
void array_scalar_div_assign_double(double *a, int size, int stride, double scalar);

#ifdef __cplusplus
}
#endif
