#pragma once

#include "zenu_compute_type.h"
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ZenuComputeConvCpu ZenuComputeConvCpu;

/**
 * @brief ZenuComputeConvCpu ハンドルの作成（Create）
 * 
 * @param[out] conv_cpu  生成されたハンドルを返すための二重ポインタ
 * 
 * ここではまだ畳み込みのパラメータは設定されていません。
 * 使用する前に @ref zenu_compute_set_conv_cpu_descriptor と
 */
ZenuStatus zenu_compute_create_conv_cpu(ZenuComputeConvCpu** conv_cpu);

/**
 * @brief ZenuComputeConvCpu に対して畳み込みパラメータを設定（Descriptor のセット）
 * 
 * @param[in] conv_cpu        ハンドル (create 済み)
 * @param[in] input           入力データの次元 2Dの場合は(N, C, H, W)
 * @param[in] output          出力データの次元 2Dの場合は(N, K, P, Q)
 * @param[in] kernel          フィルタ (カーネル) の次元 2Dの場合は(K, C, R, S)
 * @param[in] pad[2]          パディング (pad_h, pad_w)
 * @param[in] stride[2]       ストライド (stride_h, stride_w)
 * @param[in] dilation[2]     ダイレーション (dil_h, dil_w)
 * @param[in] data_type       データ型 (ZenuDataType::f32 など)
 * @param[in] num_dim         次元数 (2D など)
 * @return ZenuStatus         Success or Error
 * 
 */
ZenuStatus zenu_compute_set_conv_cpu_descriptor_cpu(
    ZenuComputeConvCpu* conv_cpu,
    size_t *input,
    size_t *output,
    size_t *kernel,
    size_t *pad,
    size_t *stride,
    size_t *dilation,
    ZenuDataType data_type,
    size_t num_dim
);

/**
 * @brief forward conv の実行に必要なワークスペース (バイト数) を取得
 * 
 * @param[in] conv_cpu  有効なコンテキストへのポインタ
 * @return 必要なワークスペースのバイト数
 */
size_t zenu_compute_conv_get_forward_workspace_bytes_cpu(ZenuComputeConvCpu* conv_cpu);

/**
 * @brief forward conv 実行
 * 
 * @param[in]  conv_cpu   有効なコンテキスト
 * @param[in]  input      入力データ (NCHW 順)
 * @param[in]  kernel     フィルタ (カーネル) データ
 * @param[in]  workspace  ワークスペース（十分なサイズを確保すること）
 * @param[out] output     出力データ (NCHW 順)
 * @return ZenuStatus     成功時 Success、失敗時はエラーコード
 */
ZenuStatus zenu_compute_conv_forward_cpu(
    ZenuComputeConvCpu* conv_cpu,
    const void* input,
    const void* kernel,
    void* workspace,
    void* output
);

/**
 * @brief Conv backward (w.r.t. data) 実行に必要なワークスペース (バイト数) を取得
 */
size_t zenu_compute_conv_get_bkwd_data_workspace_bytes_cpu(ZenuComputeConvCpu* conv_cpu);

/**
 * @brief Conv backward wrt data 実行 (d_output => d_input)
 */
ZenuStatus zenu_compute_conv_backward_data_cpu(
    ZenuComputeConvCpu* conv_cpu,
    const void* d_output,
    const void* kernel,
    void* workspace,
    void* d_input
);

/**
 * @brief Conv backward (w.r.t. kernel) 実行に必要なワークスペース (バイト数)
 */
size_t zenu_compute_conv_get_bkwd_kernel_workspace_bytes_cpu(ZenuComputeConvCpu* conv_cpu);

/**
 * @brief Conv backward wrt kernel 実行 (d_output & input => d_kernel)
 */
ZenuStatus zenu_compute_conv_backward_kernel_cpu(
    ZenuComputeConvCpu* conv_cpu,
    const void* d_output,
    const void* input,
    void* workspace,
    void* d_kernel
);

/**
 * @brief ZenuComputeConvCpu の破棄 (Destroy)
 * 
 * @param[in,out] conv_cpu  ハンドル。処理後は無効化される。
 */
void zenu_compute_destroy_conv_cpu(ZenuComputeConvCpu* conv_cpu);

typedef struct ZenuComputeConvNvidia ZenuComputeConvNvidia;

/**
 * @brief ZenuComputeConvNvidia ハンドルの作成
 */
ZenuStatus zenu_compute_create_conv_nvidia(ZenuComputeConvNvidia** conv_nvidia);

/**
 * @brief ZenuComputeConvNvidia に対してパラメータ設定
 *
 * @param[in] input           入力データの次元 2Dの場合は(N, C, H, W)
 * @param[in] output          出力データの次元 2Dの場合は(N, K, P, Q)
 * @param[in] kernel          フィルタ (カーネル) の次元 2Dの場合は(K, C, R, S)
 * @param[in] pad[2]          パディング (pad_h, pad_w)
 * @param[in] stride[2]       ストライド (stride_h, stride_w)
 * @param[in] dilation[2]     ダイレーション (dil_h, dil_w)
 * @param[in] data_type       データ型 (ZenuDataType::f32 など)
 * @param[in] num_dim         次元数 (2D など)
 * @return ZenuStatus         Success or Error
 */
ZenuStatus zenu_compute_set_conv_nvidia_descriptor(
    ZenuComputeConvNvidia* conv_nvidia,
    size_t* input,
    size_t* output,
    size_t* kernel,
    size_t stride[2],
    size_t pad[2],
    size_t dilation[2],
    ZenuDataType data_type,
    size_t num_dim
);

/**
 * @brief forward conv の実行に必要なワークスペース (Nvidia)
 */
size_t zenu_compute_conv_get_forward_workspace_bytes_nvidia(ZenuComputeConvNvidia* conv_nvidia);

/**
 * @brief forward conv 実行 (Nvidia)
 */
ZenuStatus zenu_compute_conv_forward_nvidia(
    ZenuComputeConvNvidia* conv_nvidia,
    const void* input,
    const void* kernel,
    void* workspace,
    void* output
);

/**
 * @brief backward wrt data のワークスペース取得 (Nvidia)
 */
size_t zenu_compute_conv_get_bkwd_data_workspace_bytes_nvidia(ZenuComputeConvNvidia* conv_nvidia);

/**
 * @brief backward wrt data (Nvidia)
 */
ZenuStatus zenu_compute_conv_backward_data_nvidia(
    ZenuComputeConvNvidia* conv_nvidia,
    const void* d_output,
    const void* kernel,
    void* workspace,
    void* d_input
);

/**
 * @brief backward wrt kernel のワークスペース取得 (Nvidia)
 */
size_t zenu_compute_conv_get_bkwd_kernel_workspace_bytes_nvidia(ZenuComputeConvNvidia* conv_nvidia);

/**
 * @brief backward wrt kernel (Nvidia)
 */
ZenuStatus zenu_compute_conv_backward_kernel_nvidia(
    ZenuComputeConvNvidia* conv_nvidia,
    const void* d_output,
    const void* input,
    void* workspace,
    void* d_kernel
);

/**
 * @brief ZenuComputeConvNvidia の破棄
 */
void zenu_compute_destroy_conv_nvidia(ZenuComputeConvNvidia* conv_nvidia);

#ifdef __cplusplus
} // extern "C"
#endif

