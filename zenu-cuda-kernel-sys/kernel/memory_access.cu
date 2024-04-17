#include "memory_access.h"
#include <cuda_runtime.h>

void memory_access_float(float *array, int offset, float *result) {
    cudaMemcpy(result, array + offset, sizeof(float), cudaMemcpyDeviceToHost);
}

void memory_access_double(double *array, int offset, double *result) {
    cudaMemcpy(result, array + offset, sizeof(double), cudaMemcpyDeviceToHost);
}

void memory_set_float(float *array, int offset, float value) {
    cudaMemcpy(array + offset, &value, sizeof(float), cudaMemcpyHostToDevice);
}

void memory_set_double(double *array, int offset, double value) {
    cudaMemcpy(array + offset, &value, sizeof(double), cudaMemcpyHostToDevice);
}
