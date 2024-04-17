#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void memory_access_float(float *array, int offset, float *result);
void memory_access_double(double *array, int offset, double *result);

void memory_set_float(float *array, int offset, float value);
void memory_set_double(double *array, int offset, double value);

#ifdef __cplusplus
}
#endif
