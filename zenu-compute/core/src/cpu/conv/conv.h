#pragma once
#include "interface/conv.h"

/**
 * @brief ConvCpu は CPU で動作する 2D Convolution 演算の実装.
 *
 * - 入力/出力のメモリレイアウトは [N, C, H, W]
 * - カーネル (フィルタ) の形状は [channels_out_, channels_in_, kH, kW]
 * - padding, stride, dilation は [H, W] の順で指定
 */

class ConvCpu : public IConv2D {
public:
    /**
     * @brief 2D Convolution のコンストラクタです。
     * 
     * @param batch_size    バッチサイズ
     * @param channels_in   入力チャネル数
     * @param channels_out  出力チャネル数
     * @param input_shape   入力データの形状 (H, W)
     * @param output_shape  出力データの形状 (H, W)
     * @param kernel_shape  カーネルの形状 (kH, kW)
     * @param pad           パディング (padH, padW)
     * @param stride        ストライド (strideH, strideW)
     */
    ConvCpu(size_t batch_size,
            size_t channels_in,
            size_t channels_out,
            std::vector<size_t> input_shape,
            std::vector<size_t> output_shape,
            std::vector<size_t> kernel_shape,
            std::vector<size_t> pad,
            std::vector<size_t> stride);

    virtual ~ConvCpu() = default;

    /**
     * @brief Forward 演算に必要なワークスペースサイズ (バイト単位) を取得します。
     * 
     * @return size_t ワークスペースサイズ (バイト数)。
     */
    virtual size_t get_forward_workspace_bytes() const override;

    /**
     * @brief Forward 演算を実行します (Conv2D)。
     * 
     * @param[in]  input     入力データ (NCHW)。形状: [batch_size_, channels_in_, H, W]
     * @param[in]  kernel    カーネル (フィルタ) データ。形状: [channels_out_, channels_in_, kH, kW]
     * @param[out] workspace Forward 用のワークスペース。 get_forward_workspace_bytes() のサイズ以上を確保してください。
     * @param[out] output    出力データ (NCHW)。形状: [batch_size_, channels_out_, outH, outW]
     * @return ZenuStatus    成功かエラーコードなど。
     */
    virtual ZenuStatus forward(
        const void* input,
        const void* kernel,
        void*       workspace,
        void*       output
    ) override;

    /**
     * @brief Backward-Data (入力側の勾配) 演算に必要なワークスペースサイズを取得します (バイト単位)。
     * 
     * @return size_t ワークスペースサイズ (バイト数)。
     */
    virtual size_t get_backward_data_workspace_bytes() const override;

    /**
     * @brief Backward-Data (入力側の勾配) 演算を実行します。
     * 
     * @param[in]  d_output  出力の勾配 (NCHW)。形状: [batch_size_, channels_out_, outH, outW]
     * @param[in]  kernel    カーネル (フィルタ) データ。形状: [channels_out_, channels_in_, kH, kW]
     * @param[out] workspace Backward-Data 用のワークスペース。 get_backward_data_workspace_bytes() のサイズ以上を確保。
     * @param[out] d_input   入力の勾配 (NCHW)。形状: [batch_size_, channels_in_, H, W]
     * @return ZenuStatus    成功かエラーコードなど。
     */
    virtual ZenuStatus backward_data(
        const void* d_output,
        const void* kernel,
        void*       workspace,
        void*       d_input
    ) override;

    /**
     * @brief Backward-Filter (フィルタ側の勾配) 演算に必要なワークスペースサイズを取得します (バイト単位)。
     * 
     * @return size_t ワークスペースサイズ (バイト数)。
     */
    virtual size_t get_backward_filter_workspace_bytes() const override;

    /**
     * @brief Backward-Filter (フィルタ側の勾配) 演算を実行します。
     * 
     * @param[in]  d_output  出力の勾配 (NCHW)。形状: [batch_size_, channels_out_, outH, outW]
     * @param[in]  input     入力データ (NCHW)。形状: [batch_size_, channels_in_, H, W]
     * @param[out] workspace Backward-Filter 用のワークスペース。 get_backward_filter_workspace_bytes() のサイズ以上を確保。
     * @param[out] d_kernel  フィルタの勾配。形状: [channels_out_, channels_in_, kH, kW]
     * @return ZenuStatus    成功かエラーコードなど。
     */
    virtual ZenuStatus backward_filter(
        const void* d_output,
        const void* input,
        void*       workspace,
        void*       d_kernel
    ) override;
};
