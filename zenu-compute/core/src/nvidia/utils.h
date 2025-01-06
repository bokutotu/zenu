#pragma once

#include "zenu_compute.h"
#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// (例) CUDAエラーを ZenuStatus に変換するユーティリティ関数
//      （本プロジェクトですでに用意されているならそちらを使用）
//------------------------------------------------------------------------------
static inline ZenuStatus convertCudaError(cudaError_t err)
{
    if (err == cudaSuccess) {
        return Success;
    } else {
        return DeviceError;
    }
}
