#pragma once

#include "zenu_compute_type.h"

#include <array>
#include <cstddef>
#include <vector>

/**
 * @brief CPU上で畳み込み演算を実行するクラス
 * 
 * NCHWデータレイアウトに対応した2D/3D畳み込み演算を提供します。
 * im2col + GEMM方式を採用し、OpenMPによる並列化を実装しています。
 */
struct ZenuComputeConvCpuImpl {
public:
    /**
     * @brief コンストラクタ
     */
    ZenuComputeConvCpuImpl() {};
    
    /**
     * @brief デストラクタ
     */
    ~ZenuComputeConvCpuImpl() {};

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
                    ZenuDataType type) {
        this->input = input;
        this->output = output;
        this->kernel = kernel;
        this->stride = stride;
        this->padding = padding;
        this->dilation = dilation;
        this->type = type;

        if (input.size() != output.size() || input.size() != kernel.size()) {
            return InvalidArgument;
        }
        if (stride.size() != input.size() - 2 || padding.size() != input.size() - 2 || dilation.size() != input.size() - 2) {
            return InvalidArgument;
        }
        return Success;
    }

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
    ZenuStatus forward(const void* input, const void* kernel, void* output, void* workspace) const;
    
        /**
         * @brief 逆伝搬処理（入力勾配計算）
         * 
         * @param kernel カーネルデータポインタ
         * @param grad_output 出力勾配ポインタ
         * @param grad_input 入力勾配ポインタ（事前にメモリ確保が必要）
         * @param workspace ワークスペース用バッファ
         * @return ZenuStatus 実行結果ステータス
         */
    ZenuStatus backward_data(const void* kernel, const void* grad_output, void* grad_input, void* workspace) const;
    
        /**
         * @brief 逆伝搬処理（カーネル勾配計算）
         * 
         * @param input 入力データポインタ
         * @param grad_output 出力勾配ポインタ
         * @param grad_kernel カーネル勾配ポインタ（事前にメモリ確保が必要）
         * @param workspace ワークスペース用バッファ
         * @return ZenuStatus 実行結果ステータス
         */
    ZenuStatus backward_kernel(const void* input, const void* grad_output, void* grad_kernel, void* workspace) const;
private:
    std::vector<size_t> input;   ///< 入力テンソル次元
    std::vector<size_t> output;  ///< 出力テンソル次元
    std::vector<size_t> kernel;  ///< カーネル次元
    std::vector<size_t> stride;  ///< ストライド
    std::vector<size_t> padding; ///< パディング
    std::vector<size_t> dilation;///< 拡張率
    ZenuDataType type;           ///< データ型

    /**
     * @brief 畳み込みの次元数を取得（2D=2, 3D=3）
     * @return size_t 次元数
     */
    size_t get_dim() const { return input.size() - 2; }

    /**
     * @brief im2col処理に必要なワークスペースサイズを計算
     * @return size_t 必要なバイト数
     */
    size_t get_im2col_bytes() const ;

    /**
     * @brief 1D畳み込み用im2colワークスペースサイズを計算
     * @return size_t 必要なバイト数
     */
    size_t get_im2col1d_bytes() const ;

    /**
     * @brief 2D畳み込み用im2colワークスペースサイズを計算
     * @return size_t 必要なバイト数
     */
    size_t get_im2col2d_bytes() const ;

     /**
     * @brief GEMM操作に必要なワークスペースサイズを計算 for forward
     * @return size_t 必要なバイト数
     */
    size_t get_gemm_bytes_fwd() const;

    /**
     * @brief GEMM操作に必要なワークスペースサイズを計算 for backward data
     * @return size_t 必要なバイト数
     */
    size_t get_gemm_bytes_bkwd_data() const;

    /**
     * @brief GEMM用パラメータ(M,K,N)を取得 for forward
     * @return std::array<size_t,3> [M, K, N]の配列
     */
    std::array<size_t, 3> get_gemm_param_fwd() const;

    /**
     * @brief 1D畳み込み用GEMMパラメータ(M,K,N)を取得 for forward
     * @return std::array<size_t,3> [M, K, N]の配列
     * @note M:出力チャネル数, K:入力チャネル×カーネルサイズ, N:出力空間サイズ
     */
    std::array<size_t, 3> get_gemm_param1d_fwd() const;

    /**
     * @brief 2D畳み込み用GEMMパラメータ(M,K,N)を取得 for forward
     * @return std::array<size_t,3> [M, K, N]の配列
     * @note M:出力チャネル数, K:入力チャネル×カーネルサイズ, N:出力空間サイズ
     */
    std::array<size_t, 3> get_gemm_param2d_fwd() const;

    /**
     * @brief GEMM用パラメータ(M,K,N)を取得 for backward data
     * @return std::array<size_t,3> [M, K, N]の配列
     */
    std::array<size_t, 3> get_gemm_param_bkwd_data() const;

    /**
     * @brief GEMM用パラメータ(M,K,N)を取得 for backward data
     * @return std::array<size_t,3> [M, K, N]の配列
     */
    std::array<size_t, 3> get_gemm_param1d_bkwd_data() const;

    /**
     * @brief GEMM用パラメータ(M,K,N)を取得 for backward data
     * @return std::array<size_t,3> [M, K, N]の配列
     */
    std::array<size_t, 3> get_gemm_param2d_bkwd_data() const;

        /**
     * @brief backward kernel用 GEMMパラメータ(M,K,N)を取得
     * @return [M, K, N] の配列
     */
    std::array<size_t, 3> get_gemm_param_bkwd_kernel() const;

    /**
     * @brief 2D畳み込み用 backward kernel GEMMパラメータ(M,K,N)
     */
    std::array<size_t, 3> get_gemm_param2d_bkwd_kernel() const;

    /**
     * @brief 入力テンソルをcolumn行列に変換（im2col）
     * @param input 入力データポインタ
     * @param col 出力column行列ポインタ
     */
    void im2col(const void* input, void* col) const;

    /**
     * @brief 1D畳み込み用im2col処理
     * @param input 入力データポインタ
     * @param col 出力column行列ポインタ
     * @note OpenMPによる並列化済み
     */
    void im2col1d(const void* input, void* col)const;
    
    /**
     * @brief 2D畳み込み用im2col処理
     * @param input 入力データポインタ
     * @param col 出力column行列ポインタ
     * @note OpenMPによる並列化済み
     */
    void im2col2d(const void* input, void* col)const;

    /**
     * @brief column行列を入力テンソル形式に変換（col2im）
     * @param col 入力column行列ポインタ
     * @param input 出力テンソルポインタ
     */
    void col2im(const void* col, void* input) const;

    /**
     * @brief 1D畳み込み用col2im処理
     * @param col 入力column行列ポインタ
     * @param input 出力テンソルポインタ
     * @note OpenMPによる並列化済み
     */
    void col2im1d(const void* col, void* input) const;

    /**
     * @brief 2D畳み込み用col2im処理
     * @param col 入力column行列ポインタ
     * @param input 出力テンソルポインタ
     * @note OpenMPによる並列化済み
     */
    void col2im2d(const void* col, void* input) const;

    /**
     * @brief GEMM出力を適切なレイアウトに転置 for forward
     * @param gemm_out GEMM出力ポインタ
     * @param output 転置済み出力ポインタ
     */
    void transpose_gemm_fwd(const void* gemm_out, void* output) const;

    /**
     * @brief 1D畳み込み用GEMM出力転置処理 for forward
     * @param gemm_out GEMM出力ポインタ
     * @param output 転置済み出力ポインタ
     * @note NCHWレイアウトに適合するよう4次元ループで転置
     */
    void transpose_gemm1d_fwd(const void* gemm_out, void* output) const;

    /**
     * @brief 2D畳み込み用GEMM出力転置処理 for forward
     * @param gemm_out GEMM出力ポインタ
     * @param output 転置済み出力ポインタ
     * @note NCHWレイアウトに適合するよう4次元ループで転置
     */
    void transpose_gemm2d_fwd(const void* gemm_out, void* output) const;

    /**
     * @brief GEMM出力を適切なレイアウトに転置 for backward data
     * @param d_output GEMM出力ポインタ
     * @param d_output_reshaped 転置済み出力ポインタ
     */
    void transpose_gemm_bkwd_data(const void* d_output, void* d_output_reshaped) const;

    /**
     * @brief GEMM出力を適切なレイアウトに転置 for backward kernel
     * @param d_output GEMM出力ポインタ
     * @param d_output_reshaped 転置済み出力ポインタ
     */
    void transpose_gemm1d_bkwd_data(const void* d_output, void* d_output_reshaped) const;

    /**
     * @brief GEMM出力を適切なレイアウトに転置 for backward kernel
     * @param d_output GEMM出力ポインタ
     * @param d_output_reshaped 転置済み出力ポインタ
     */
    void transpose_gemm2d_bkwd_data(const void* d_output, void* d_output_reshaped) const;

    /**
     * @brief GEMM出力を適切なレイアウトに転置 for backward kernel
     * @param grad_output GEMM出力ポインタ
     * @param grad_output_reshaped 転置済み出力ポインタ
     */
    void transpose_gemm_bkwd_kernel(const void* grad_output, void* grad_output_reshaped) const;

    /**
     * @brief 2D畳み込み用 grad_output 転置 ( [N,K,PQ] => [K, N*PQ] )
     */
    void transpose_gemm2d_bkwd_kernel(
        const void* grad_output,
        void*       grad_output_reshaped
    ) const;
};
