#pragma once

#include "zenu_compute_type.h"
#include <cuda_runtime.h>

static inline ZenuStatus convertCudaError(cudaError_t err)
{
    if (err == cudaSuccess) {
        return Success;
    } else {
        return DeviceError;
    }
}
