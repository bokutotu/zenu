#pragma once

#include "zenu_compute_type.h"
#include "nvidia/cudnn/graph_desc_interface.h"
#include <memory>

struct ZenuComputeConvNvidiaImpl {
public:
    ZenuComputeConvNvidiaImpl() {};
    ~ZenuComputeConvNvidiaImpl() {};

    /**
     * @brief 畳み込みパラメータを初期化
     * 
     * @param input 入力テンソルの次元 [N, C, H, W] (2Dの場合)
     * @param output 出力テンソルの次元 [N, K, P, Q] (2Dの場合)
     * @param kernel カーネルサイズ [K, C, R, S] (2Dの場合)
     * @param stride ストライド [stride_h, stride_w]
     * @param padding パディング [pad_h, pad_w]
     * @param dilation 拡張率 [dilation_h, dilation_w]
     * @param type データ型 (f32/f64)
     * @return ZenuStatus 初期化結果ステータス
     * @throws std::invalid_argument パラメータが不正な場合
     */
    ZenuStatus init(std::vector<size_t> input, 
                    std::vector<size_t> output, 
                    std::vector<size_t> kernel, 
                    std::vector<size_t> stride, 
                    std::vector<size_t> padding, 
                    std::vector<size_t> dilation,
                    ZenuDataType type);

    /**
     * @brief 順伝搬の出力テンソルに必要なメモリ量を計算
     * @return size_t 必要なバイト数
     */
    size_t get_forward_bytes() const;

    /**
     * @brief 逆伝搬（データ勾配）に必要なメモリ量を計算
     * @return size_t 必要なバイト数
     */
    size_t get_backward_data_bytes() const;

    /**
     * @brief 逆伝搬（カーネル勾配）に必要なメモリ量を計算
     * @return size_t 必要なバイト数
     */
    size_t get_backward_kernel_bytes() const;

    /**
     * @brief 順伝搬処理（畳み込み演算）
     * 
     * @param input 入力データポインタ
     * @param kernel カーネルデータポインタ
     * @param output 出力データポインタ（事前にメモリ確保が必要）
     * @param workspace ワークスペース用バッファ
     * @return ZenuStatus 実行結果ステータス
     */
    ZenuStatus forward(const void* input, 
                       const void* kernel, 
                       void* output, 
                       void* workspace) const ;

    /**
     * @brief 逆伝搬処理（入力勾配計算）
     *
     * @param kernel カーネルデータポインタ
     * @param grad_output 出力勾配ポインタ
     * @param grad_input 入力勾配ポインタ（事前にメモリ確保が必要）
     * @param workspace ワークスペース用バッファ
     * @return ZenuStatus 実行結果ステータス
     */
    ZenuStatus backward_data(const void* kernel, 
                             const void* grad_output, 
                             void* grad_input, 
                             void* workspace) const ;

    /**
     * @brief 逆伝搬処理（カーネル勾配計算）
     * 
     * @param input 入力データポインタ
     * @param grad_output 出力勾配ポインタ
     * @param grad_kernel カーネル勾配ポインタ（事前にメモリ確保が必要）
     * @param workspace ワークスペース用バッファ
     * @return ZenuStatus 実行結果ステータス
     */
    ZenuStatus backward_kernel(const void* input, 
                               const void* grad_output, 
                               void* grad_kernel, 
                               void* workspace) const;

private:
    std::vector<size_t> input;   ///< 入力テンソル次元
    std::vector<size_t> output;  ///< 出力テンソル次元
    std::vector<size_t> kernel;  ///< カーネル次元
    std::vector<size_t> stride;  ///< ストライド
    std::vector<size_t> padding; ///< パディング
    std::vector<size_t> dilation;///< 拡張率
    ZenuDataType type;           ///< データ型

    std::shared_ptr<fe::graph::Graph> fwd_graph;
    std::shared_ptr<fe::graph::Graph> bwd_data_graph;
    std::shared_ptr<fe::graph::Graph> bwd_kernel_graph;

    std::shared_ptr<fe::graph::Tensor_attributes> X_fwd;
    std::shared_ptr<fe::graph::Tensor_attributes> Kernel_fwd;
    std::shared_ptr<fe::graph::Tensor_attributes> Y_fwd;

    fe::graph::Conv_fprop_attributes conv_options;


    ZenuStatus init_fwd();
    ZenuStatus init_bwd_data();
    ZenuStatus init_bwd_kernel();
};
