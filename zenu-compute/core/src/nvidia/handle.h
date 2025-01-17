#pragma once

#include <cublas_v2.h>
#include <cudnn.h>
#include <stdexcept>

/**
 * スレッド毎に cublasHandle_t / cudnnHandle_t を保持するクラス。
 * 下記の静的メソッドを通して取得すると、スレッドローカルなハンドルが返される。
 *
 * - NvidiaHandles::getCublasHandle()
 * - NvidiaHandles::getCudnnHandle()
 */
class NvidiaHandles {
public:
    /**
     * @brief 現在のスレッドに割り当てられた cublasHandle_t を返す。
     *        初回呼び出し時に cublasCreate_v2() で生成し、スレッド終了時に自動解放される。
     */
    static cublasHandle_t getCublasHandle() {
        thread_local static CublasHandleWrapper cublasWrapper;
        return cublasWrapper.handle;
    }

    /**
     * @brief 現在のスレッドに割り当てられた cudnnHandle_t を返す。
     *        初回呼び出し時に cudnnCreate() で生成し、スレッド終了時に自動解放される。
     */
    static cudnnHandle_t getCudnnHandle() {
        thread_local static CudnnHandleWrapper cudnnWrapper;
        return cudnnWrapper.handle;
    }

private:
    /**
     * cublasHandle_t をスレッドローカルに保持するラッパ。
     * コンストラクタで cublasCreate()、デストラクタで cublasDestroy() を呼ぶ。
     */
    struct CublasHandleWrapper {
        cublasHandle_t handle;

        CublasHandleWrapper() {
            cublasStatus_t stat = cublasCreate_v2(&handle);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create cublas handle.");
            }
        }

        ~CublasHandleWrapper() {
            // スレッド終了時に自動解放される。
            cublasDestroy_v2(handle);
        }
    };

    /**
     * cudnnHandle_t をスレッドローカルに保持するラッパ。
     * コンストラクタで cudnnCreate()、デストラクタで cudnnDestroy() を呼ぶ。
     */
    struct CudnnHandleWrapper {
        cudnnHandle_t handle;

        CudnnHandleWrapper() {
            cudnnStatus_t stat = cudnnCreate(&handle);
            if (stat != CUDNN_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create cudnn handle.");
            }
        }

        ~CudnnHandleWrapper() {
            // スレッド終了時に自動解放される。
            cudnnDestroy(handle);
        }
    };
};

