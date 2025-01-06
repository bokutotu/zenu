#pragma once
#include <cublas_v2.h>
#include <cudnn.h>

class NvidiaHandles {
public:
    cublasHandle_t* cublasHandle;
    cudnnHandle_t*  cudnnHandle;

    NvidiaHandles() {
        cublasCreate_v2(this->cublasHandle);
        cudnnCreate(this->cudnnHandle);
    }

    ~NvidiaHandles() {
        cublasDestroy_v2(*this->cublasHandle);
        cudnnDestroy(*this->cudnnHandle);
    }
};

// グローバル変数的に扱う関数
inline NvidiaHandles& get_global_nvidia_handles() {
    static NvidiaHandles instance;
    return instance;
}
