#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void array_max_idx_float(float *a, int size, int stride, int *out);

void array_max_idx_double(double *a, int size, int stride, int *out);

#ifdef __cplusplus
 }
#endif
