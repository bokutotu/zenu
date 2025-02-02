#pragma once

#include "zenu_compute_type.h"
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief ZenuComputeConvCpu 構造体の前方宣言
 *
 * この構造体は、CPU を用いた畳み込み演算のためのハンドルを管理するために使用されます。
 */
typedef struct ZenuComputeConvCpu ZenuComputeConvCpu;

/**
 * @brief ZenuComputeConvCpu ハンドルの作成（Create）
 *
 * CPU 上での畳み込み演算に使用するハンドルを生成します。<br>
 * ※ 作成時点では畳み込みパラメータは設定されていません。<br>
 *    使用前に @ref zenu_compute_set_conv_cpu_descriptor_cpu を用いてパラメータを設定してください。
 *
 * @param[out] conv_cpu  生成されたハンドルを返すためのポインタのアドレス
 * @return ZenuStatus   成功時は Success、それ以外はエラーコードを返します。
 */
ZenuStatus zenu_compute_create_conv_cpu(ZenuComputeConvCpu** conv_cpu);

/**
 * @brief ZenuComputeConvCpu に対して畳み込みパラメータを設定（Descriptor のセット）
 *
 * 畳み込み演算に必要なパラメータを設定します。<br>
 * 
 * 【2D 畳み込みの場合】<br>
 *  - 入力:  (N, C, H, W)<br>
 *  - 出力: (N, K, P, Q)<br>
 *  - カーネル: (K, C, R, S)<br>
 *  - pad:    (pad_h, pad_w)<br>
 *  - stride: (stride_h, stride_w)<br>
 *  - dilation: (dil_h, dil_w)<br>
 * 
 * 【1D 畳み込みの場合】<br>
 *  - 入力:  (N, C, L)<br>
 *  - 出力: (N, K, L_out)<br>
 *  - カーネル: (K, C, S)<br>
 *  - pad:    (pad, _)   （pad[0] のみ使用）<br>
 *  - stride: (stride, _)（stride[0] のみ使用）<br>
 *  - dilation: (dilation, _)（dilation[0] のみ使用）<br>
 *
 * @param[in] conv_cpu    @ref zenu_compute_create_conv_cpu で作成されたハンドル
 * @param[in] input       入力データの各次元サイズを格納した配列へのポインタ
 * @param[in] output      出力データの各次元サイズを格納した配列へのポインタ
 * @param[in] kernel      カーネル（フィルタ）の各次元サイズを格納した配列へのポインタ
 * @param[in] pad         パディング値の配列（1D の場合は pad[0] のみ使用）
 * @param[in] stride      ストライド値の配列（1D の場合は stride[0] のみ使用）
 * @param[in] dilation    ダイレーション値の配列（1D の場合は dilation[0] のみ使用）
 * @param[in] data_type   データ型（例: ZenuDataType::f32）
 * @param[in] num_dim     次元数（例: 1 → 1D、2 → 2D）
 * @return ZenuStatus     成功時は Success、それ以外はエラーコードを返します。
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
 * @brief forward 畳み込み実行に必要なワークスペースのサイズ（バイト数）を取得
 *
 * forward 畳み込み演算を実行するために必要なワークスペースのサイズをバイト単位で返します。<br>
 * ワークスペースサイズは、設定されたパラメータおよび次元数（1D または 2D）に依存します。
 *
 * @param[in] conv_cpu  有効な畳み込みコンテキストのハンドル
 * @return size_t       必要なワークスペースのバイト数
 */
size_t zenu_compute_conv_get_forward_workspace_bytes_cpu(ZenuComputeConvCpu* conv_cpu);

/**
 * @brief forward 畳み込みの実行（CPU）
 *
 * CPU 上で forward 畳み込み演算を実行します。<br>
 * 
 * 【2D 畳み込みの場合】<br>
 *  - 入力は NCHW 順 (N, C, H, W) であること<br>
 *  - 出力は NCHW 順 (N, K, P, Q) であること<br>
 *  - カーネルは (K, C, R, S) の形式<br>
 * 
 * 【1D 畳み込みの場合】<br>
 *  - 入力は NCW 順 (N, C, L) であること<br>
 *  - 出力は NCW 順 (N, K, L_out) であること<br>
 *  - カーネルは (K, C, S) の形式<br>
 *
 * @param[in]  conv_cpu   有効な畳み込みコンテキストのハンドル
 * @param[in]  input      入力データへのポインタ
 * @param[in]  kernel     カーネル（フィルタ）データへのポインタ
 * @param[in]  workspace  ワークスペースへのポインタ（@ref zenu_compute_conv_get_forward_workspace_bytes_cpu で取得したサイズ以上のメモリを確保すること）
 * @param[out] output     出力データへのポインタ
 * @return ZenuStatus     成功時は Success、それ以外はエラーコードを返します。
 */
ZenuStatus zenu_compute_conv_forward_cpu(
    ZenuComputeConvCpu* conv_cpu,
    const void* input,
    const void* kernel,
    void* workspace,
    void* output
);

/**
 * @brief backward 畳み込み（入力データ勾配）実行に必要なワークスペースのサイズ（バイト数）を取得
 *
 * backward 畳み込み（入力データに対する勾配計算）を実行するために必要なワークスペースのサイズをバイト単位で返します。
 *
 * @param[in] conv_cpu  有効な畳み込みコンテキストのハンドル
 * @return size_t       必要なワークスペースのバイト数
 */
size_t zenu_compute_conv_get_bkwd_data_workspace_bytes_cpu(ZenuComputeConvCpu* conv_cpu);

/**
 * @brief backward 畳み込み（入力データに対する勾配計算）の実行（CPU）
 *
 * 出力の勾配 (d_output) とカーネル情報から、入力データに対する勾配 (d_input) を計算します。<br>
 * 
 * 【2D 畳み込みの場合】<br>
 *  - d_output は NCHW 順 (N, K, P, Q) であること<br>
 *  - カーネルは (K, C, R, S) の形式<br>
 *  - d_input は NCHW 順 (N, C, H, W) であること<br>
 * 
 * 【1D 畳み込みの場合】<br>
 *  - d_output は NCW 順 (N, K, L_out) であること<br>
 *  - カーネルは (K, C, S) の形式<br>
 *  - d_input は NCW 順 (N, C, L) であること<br>
 *
 * @param[in] conv_cpu  有効な畳み込みコンテキストのハンドル
 * @param[in] d_output  出力の勾配データへのポインタ
 * @param[in] kernel    カーネル（フィルタ）データへのポインタ
 * @param[in] workspace ワークスペースへのポインタ（@ref zenu_compute_conv_get_bkwd_data_workspace_bytes_cpu で取得したサイズ以上のメモリを確保すること）
 * @param[out] d_input  入力データに対する勾配データへのポインタ
 * @return ZenuStatus   成功時は Success、それ以外はエラーコードを返します。
 */
ZenuStatus zenu_compute_conv_backward_data_cpu(
    ZenuComputeConvCpu* conv_cpu,
    const void* d_output,
    const void* kernel,
    void* workspace,
    void* d_input
);

/**
 * @brief backward 畳み込み（カーネル勾配）実行に必要なワークスペースのサイズ（バイト数）を取得
 *
 * backward 畳み込み（カーネルに対する勾配計算）を実行するために必要なワークスペースのサイズをバイト単位で返します。
 *
 * @param[in] conv_cpu  有効な畳み込みコンテキストのハンドル
 * @return size_t       必要なワークスペースのバイト数
 */
size_t zenu_compute_conv_get_bkwd_kernel_workspace_bytes_cpu(ZenuComputeConvCpu* conv_cpu);

/**
 * @brief backward 畳み込み（カーネルに対する勾配計算）の実行（CPU）
 *
 * 出力の勾配 (d_output) と入力データから、カーネルに対する勾配 (d_kernel) を計算します。<br>
 * 
 * 【2D 畳み込みの場合】<br>
 *  - d_output は NCHW 順 (N, K, P, Q) であること<br>
 *  - 入力は NCHW 順 (N, C, H, W) であること<br>
 *  - d_kernel は (K, C, R, S) の形式<br>
 * 
 * 【1D 畳み込みの場合】<br>
 *  - d_output は NCW 順 (N, K, L_out) であること<br>
 *  - 入力は NCW 順 (N, C, L) であること<br>
 *  - d_kernel は (K, C, S) の形式<br>
 *
 * @param[in] conv_cpu  有効な畳み込みコンテキストのハンドル
 * @param[in] d_output  出力の勾配データへのポインタ
 * @param[in] input     入力データへのポインタ
 * @param[in] workspace ワークスペースへのポインタ（@ref zenu_compute_conv_get_bkwd_kernel_workspace_bytes_cpu で取得したサイズ以上のメモリを確保すること）
 * @param[out] d_kernel カーネルに対する勾配データへのポインタ
 * @return ZenuStatus   成功時は Success、それ以外はエラーコードを返します。
 */
ZenuStatus zenu_compute_conv_backward_kernel_cpu(
    ZenuComputeConvCpu* conv_cpu,
    const void* d_output,
    const void* input,
    void* workspace,
    void* d_kernel
);

/**
 * @brief ZenuComputeConvCpu ハンドルの破棄（Destroy）
 *
 * CPU 畳み込みハンドルおよび関連リソースを解放します。<br>
 * 本関数実行後、ハンドルは無効となります。
 *
 * @param[in,out] conv_cpu  破棄対象のハンドルへのポインタ
 */
void zenu_compute_destroy_conv_cpu(ZenuComputeConvCpu* conv_cpu);

/**
 * @brief ZenuComputeConvNvidia 構造体の前方宣言
 *
 * この構造体は、NVIDIA GPU を用いた畳み込み演算のためのハンドルを管理するために使用されます。
 */
typedef struct ZenuComputeConvNvidia ZenuComputeConvNvidia;

/**
 * @brief ZenuComputeConvNvidia ハンドルの作成（Create）
 *
 * NVIDIA GPU 上での畳み込み演算に使用するハンドルを生成します。
 *
 * @param[out] conv_nvidia 生成されたハンドルを返すためのポインタのアドレス
 * @return ZenuStatus    成功時は Success、それ以外はエラーコードを返します。
 */
ZenuStatus zenu_compute_create_conv_nvidia(ZenuComputeConvNvidia** conv_nvidia);

/**
 * @brief ZenuComputeConvNvidia に対して畳み込みパラメータを設定（Descriptor のセット）
 *
 * 畳み込み演算に必要なパラメータを設定します。<br>
 * 
 * 【2D 畳み込みの場合】<br>
 *  - 入力:  (N, C, H, W)<br>
 *  - 出力: (N, K, P, Q)<br>
 *  - カーネル: (K, C, R, S)<br>
 *  - pad:    (pad_h, pad_w)<br>
 *  - stride: (stride_h, stride_w)<br>
 *  - dilation: (dil_h, dil_w)<br>
 * 
 * 【1D 畳み込みの場合】<br>
 *  - 入力:  (N, C, L)<br>
 *  - 出力: (N, K, L_out)<br>
 *  - カーネル: (K, C, S)<br>
 *  - pad:    (pad, _)   （pad[0] のみ使用）<br>
 *  - stride: (stride, _)（stride[0] のみ使用）<br>
 *  - dilation: (dilation, _)（dilation[0] のみ使用）<br>
 *
 * @param[in] conv_nvidia  @ref zenu_compute_create_conv_nvidia で作成されたハンドル
 * @param[in] input        入力データの各次元サイズを格納した配列へのポインタ
 * @param[in] output       出力データの各次元サイズを格納した配列へのポインタ
 * @param[in] kernel       カーネル（フィルタ）の各次元サイズを格納した配列へのポインタ
 * @param[in] pad          パディング値の配列（1D の場合は pad[0] のみ使用）
 * @param[in] stride       ストライド値の配列（1D の場合は stride[0] のみ使用）
 * @param[in] dilation     ダイレーション値の配列（1D の場合は dilation[0] のみ使用）
 * @param[in] data_type    データ型（例: ZenuDataType::f32）
 * @param[in] num_dim      次元数（例: 1 → 1D、2 → 2D）
 * @return ZenuStatus      成功時は Success、それ以外はエラーコードを返します。
 */
ZenuStatus zenu_compute_set_conv_nvidia_descriptor(
    ZenuComputeConvNvidia* conv_nvidia,
    size_t* input,
    size_t* output,
    size_t* kernel,
    size_t pad[2],
    size_t stride[2],
    size_t dilation[2],
    ZenuDataType data_type,
    size_t num_dim
);

/**
 * @brief forward 畳み込み実行に必要なワークスペースのサイズ（バイト数）を取得（Nvidia）
 *
 * NVIDIA GPU 上で forward 畳み込み演算を実行するために必要なワークスペースのサイズをバイト単位で返します。<br>
 * ワークスペースサイズは、設定されたパラメータおよび次元数（1D または 2D）に依存します。
 *
 * @param[in] conv_nvidia 有効な NVIDIA 畳み込みコンテキストのハンドル
 * @return size_t       必要なワークスペースのバイト数
 */
size_t zenu_compute_conv_get_forward_workspace_bytes_nvidia(ZenuComputeConvNvidia* conv_nvidia);

/**
 * @brief forward 畳み込みの実行（Nvidia）
 *
 * NVIDIA GPU 上で forward 畳み込み演算を実行します。<br>
 * 
 * 【2D 畳み込みの場合】<br>
 *  - 入力は NCHW 順 (N, C, H, W) であること<br>
 *  - 出力は NCHW 順 (N, K, P, Q) であること<br>
 *  - カーネルは (K, C, R, S) の形式<br>
 * 
 * 【1D 畳み込みの場合】<br>
 *  - 入力は NCW 順 (N, C, L) であること<br>
 *  - 出力は NCW 順 (N, K, L_out) であること<br>
 *  - カーネルは (K, C, S) の形式<br>
 *
 * @param[in]  conv_nvidia 有効な NVIDIA 畳み込みコンテキストのハンドル
 * @param[in]  input       入力データへのポインタ
 * @param[in]  kernel      カーネル（フィルタ）データへのポインタ
 * @param[in]  workspace   ワークスペースへのポインタ（@ref zenu_compute_conv_get_forward_workspace_bytes_nvidia で取得したサイズ以上のメモリを確保すること）
 * @param[out] output      出力データへのポインタ
 * @return ZenuStatus      成功時は Success、それ以外はエラーコードを返します。
 */
ZenuStatus zenu_compute_conv_forward_nvidia(
    ZenuComputeConvNvidia* conv_nvidia,
    const void* input,
    const void* kernel,
    void* workspace,
    void* output
);

/**
 * @brief backward 畳み込み（入力データ勾配）実行に必要なワークスペースのサイズ（バイト数）を取得（Nvidia）
 *
 * NVIDIA GPU 上で backward 畳み込み（入力データに対する勾配計算）を実行するために必要なワークスペースのサイズをバイト単位で返します。
 *
 * @param[in] conv_nvidia 有効な NVIDIA 畳み込みコンテキストのハンドル
 * @return size_t       必要なワークスペースのバイト数
 */
size_t zenu_compute_conv_get_bkwd_data_workspace_bytes_nvidia(ZenuComputeConvNvidia* conv_nvidia);

/**
 * @brief backward 畳み込み（入力データに対する勾配計算）の実行（Nvidia）
 *
 * 出力の勾配 (d_output) とカーネル情報から、入力データに対する勾配 (d_input) を計算します。<br>
 * 
 * 【2D 畳み込みの場合】<br>
 *  - d_output は NCHW 順 (N, K, P, Q) であること<br>
 *  - カーネルは (K, C, R, S) の形式<br>
 *  - d_input は NCHW 順 (N, C, H, W) であること<br>
 * 
 * 【1D 畳み込みの場合】<br>
 *  - d_output は NCW 順 (N, K, L_out) であること<br>
 *  - カーネルは (K, C, S) の形式<br>
 *  - d_input は NCW 順 (N, C, L) であること<br>
 *
 * @param[in] conv_nvidia 有効な NVIDIA 畳み込みコンテキストのハンドル
 * @param[in] d_output    出力の勾配データへのポインタ
 * @param[in] kernel      カーネル（フィルタ）データへのポインタ
 * @param[in] workspace   ワークスペースへのポインタ（@ref zenu_compute_conv_get_bkwd_data_workspace_bytes_nvidia で取得したサイズ以上のメモリを確保すること）
 * @param[out] d_input    入力データに対する勾配データへのポインタ
 * @return ZenuStatus     成功時は Success、それ以外はエラーコードを返します。
 */
ZenuStatus zenu_compute_conv_backward_data_nvidia(
    ZenuComputeConvNvidia* conv_nvidia,
    const void* d_output,
    const void* kernel,
    void* workspace,
    void* d_input
);

/**
 * @brief backward 畳み込み（カーネル勾配）実行に必要なワークスペースのサイズ（バイト数）を取得（Nvidia）
 *
 * NVIDIA GPU 上で backward 畳み込み（カーネルに対する勾配計算）を実行するために必要なワークスペースのサイズをバイト単位で返します。
 *
 * @param[in] conv_nvidia 有効な NVIDIA 畳み込みコンテキストのハンドル
 * @return size_t       必要なワークスペースのバイト数
 */
size_t zenu_compute_conv_get_bkwd_kernel_workspace_bytes_nvidia(ZenuComputeConvNvidia* conv_nvidia);

/**
 * @brief backward 畳み込み（カーネルに対する勾配計算）の実行（Nvidia）
 *
 * 出力の勾配 (d_output) と入力データから、カーネルに対する勾配 (d_kernel) を計算します。<br>
 * 
 * 【2D 畳み込みの場合】<br>
 *  - d_output は NCHW 順 (N, K, P, Q) であること<br>
 *  - 入力は NCHW 順 (N, C, H, W) であること<br>
 *  - d_kernel は (K, C, R, S) の形式<br>
 * 
 * 【1D 畳み込みの場合】<br>
 *  - d_output は NCW 順 (N, K, L_out) であること<br>
 *  - 入力は NCW 順 (N, C, L) であること<br>
 *  - d_kernel は (K, C, S) の形式<br>
 *
 * @param[in] conv_nvidia 有効な NVIDIA 畳み込みコンテキストのハンドル
 * @param[in] d_output    出力の勾配データへのポインタ
 * @param[in] input       入力データへのポインタ
 * @param[in] workspace   ワークスペースへのポインタ（@ref zenu_compute_conv_get_bkwd_kernel_workspace_bytes_nvidia で取得したサイズ以上のメモリを確保すること）
 * @param[out] d_kernel   カーネルに対する勾配データへのポインタ
 * @return ZenuStatus     成功時は Success、それ以外はエラーコードを返します。
 */
ZenuStatus zenu_compute_conv_backward_kernel_nvidia(
    ZenuComputeConvNvidia* conv_nvidia,
    const void* d_output,
    const void* input,
    void* workspace,
    void* d_kernel
);

/**
 * @brief ZenuComputeConvNvidia ハンドルの破棄（Destroy）
 *
 * NVIDIA GPU 畳み込みハンドルおよび関連リソースを解放します。<br>
 * 本関数実行後、ハンドルは無効となります。
 *
 * @param[in,out] conv_nvidia  破棄対象のハンドルへのポインタ
 */
void zenu_compute_destroy_conv_nvidia(ZenuComputeConvNvidia* conv_nvidia);

#ifdef __cplusplus
} // extern "C"
#endif

