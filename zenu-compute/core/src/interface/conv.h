#pragma once

#include "zenu_compute_type.h"
#include <vector>
#include <cstddef>

/**
 * @brief IConv2D は 2D Convolution 演算のための抽象クラスです (NCHW レイアウトを仮定)。
 *
 * - 入力/出力のメモリレイアウトは [N, C, H, W] となります。
 * - カーネル (フィルタ) の形状は [channels_out_, channels_in_, kH, kW] を想定。
 * - padding, stride, dilation は [H, W] の順で指定。
 *
 * Forward (順伝搬)、Backward-Data (入力側の勾配)、Backward-Filter (フィルタ側の勾配) それぞれに必要な
 * ワークスペースサイズを取得するメソッドと演算を行うメソッドを提供します。
 */
class IConv2D {
public:
    virtual ~IConv2D() = default;

    /**
     * @brief Forward 演算に必要なワークスペースサイズ (バイト単位) を取得します。
     * 
     * @return size_t ワークスペースサイズ (バイト数)。
     */
    virtual size_t get_forward_workspace_bytes() const = 0;

    /**
     * @brief Forward 演算を実行します (Conv2D)。
     * 
     * @param[in]  input     入力データ (NCHW)。形状: [batch_size_, channels_in_, H, W]
     * @param[in]  kernel    カーネル (フィルタ) データ。形状: [channels_out_, channels_in_, kH, kW]
     * @param[out] workspace Forward 用のワークスペース。 get_forward_workspace_bytes() のサイズ以上を確保してください。
     * @param[out] output    出力データ (NCHW)。形状: [batch_size_, channels_out_, outH, outW]
     * @return ZenuStatus    成功かエラーコードなど。
     *
     * @note outH, outW は推論される出力の空間次元 (H, W)。init() で指定されたものと一致します。
     */
    virtual ZenuStatus forward(
        const void* input,
        const void* kernel,
        void*       workspace,
        void*       output
    ) = 0;

    /**
     * @brief Backward-Data (入力側の勾配) 演算に必要なワークスペースサイズを取得します (バイト単位)。
     * 
     * @return size_t ワークスペースサイズ (バイト数)。
     */
    virtual size_t get_backward_data_workspace_bytes() const = 0;

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
    ) = 0;

    /**
     * @brief Backward-Filter (フィルタ側の勾配) 演算に必要なワークスペースサイズを取得します (バイト単位)。
     * 
     * @return size_t ワークスペースサイズ (バイト数)。
     */
    virtual size_t get_backward_filter_workspace_bytes() const = 0;

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
    ) = 0;

protected:
    size_t batch_size   = 0;  ///< バッチサイズ (N)
    size_t channels_in  = 0;  ///< 入力チャネル数 (C_in)
    size_t channels_out = 0;  ///< 出力チャネル数 (C_out)

    /// 入力の空間次元 (例: [H, W])
    std::vector<size_t> input_shape;
    /// 出力の空間次元 (例: [outH, outW])
    std::vector<size_t> output_shape;
    /// カーネルサイズ (例: [kH, kW])
    std::vector<size_t> kernel_shape;
    /// パディング (例: [padH, padW])
    std::vector<size_t> pad;
    /// ストライド (例: [strideH, strideW])
    std::vector<size_t> stride;
    /// ダイレーション (例: [dilationH, dilationW])
    std::vector<size_t> dilation;

private:
    /**
     * @brief 2D Convolution のパラメータを検証します。
     * 
     * @return ZenuStatus 成功かエラーコードなど。
     */
    ZenuStatus validate() {
        if (input_shape.size() ==2) {
            return validate2d_params();
        } else {
            return ZenuStatus::InvalidShape;
        }
    }

    /**
     * @brief 2D Convolution の出力次元を計算します。
     * 
     * @param input_size   入力次元 (H or W)
     * @param kernel_size  カーネル次元 (kH or kW)
     * @param pad          パディング
     * @param stride       ストライド
     * @param dilation     ダイレーション
     * @return size_t      出力次元
     */
    size_t cal_output_shape(size_t input_size, size_t kernel_size, size_t pad, size_t stride, size_t dilation) {
        return (input_size + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1;
    }

    /**
     * @brief 2D Convolution のパラメータを検証します。
     * 
     * @return ZenuStatus 成功かエラーコードなど。
     */
    ZenuStatus validate2d_params() {
        if (input_shape.size() != 2 || output_shape.size() != 2 || kernel_shape.size() != 2 ||
            pad.size() != 2 || stride.size() != 2 || dilation.size() != 2) {
            return ZenuStatus::InvalidShape;
        }

        if (channels_in == 0 || channels_out == 0) {
            return ZenuStatus::InvalidShape;
        }

        auto output_w_expected = cal_output_shape(input_shape[1], kernel_shape[1], pad[1], stride[1], dilation[1]);
        auto output_h_expected = cal_output_shape(input_shape[0], kernel_shape[0], pad[0], stride[0], dilation[0]);

        if (output_shape[0] != output_h_expected || output_shape[1] != output_w_expected) {
            return ZenuStatus::InvalidShape;
        }
        return ZenuStatus::Success;
    }
};

