#pragma once

#include "zenu_compute_type.h"

#include <vector>
#include <cstddef>
#include <memory>
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

struct ZenuComputeBatchNormImpl {
public:
    ZenuComputeBatchNormImpl() = default;
    ~ZenuComputeBatchNormImpl() = default;

    /**
     * @brief バッチノルムのグラフを初期化する。
     * 
     * @param shape 入力テンソルの次元（例: {N, C, H, W}）
     * @param dim   バッチノルムを適用するチャネル軸（例: C 軸なら 1）
     * @param type  入出力データ型（例: ZenuDataType::HALF 等）
     * @param is_train 学習時か否か（true: training，false: inference）
     * @return ZenuStatus 初期化結果
     */
    ZenuStatus init(std::vector<size_t> shape, size_t dim, ZenuDataType type, bool is_train);

    /**
     * @brief 学習時の順伝搬（BN training）を実行する。
     * 入力 x, スケール scale, バイアス bias から出力 y を計算し、
     * バッチごとの平均 mean と逆分散 inv_variance を出力する。
     *
     * @param x 入力データポインタ
     * @param scale スケールパラメータ（1×C×1×1 など）
     * @param bias バイアスパラメータ（1×C×1×1 など）
     * @param y 出力データポインタ
     * @param mean 出力のバッチ平均ポインタ
     * @param inv_variance 出力のバッチ逆分散ポインタ
     * @param workspace ワークスペースバッファ
     * @return ZenuStatus 実行結果
     */
    ZenuStatus forward_train(const void* x, 
                             const void* scale, 
                             const void* bias, 
                             void* y, 
                             void* mean, 
                             void* inv_variance,
                             void* workspace);

    /**
     * @brief 推論時の順伝搬（BN inference）を実行する。
     * 学習時に得た mean, inv_variance を利用して正規化する。
     *
     * @param x 入力データポインタ
     * @param scale スケールパラメータ
     * @param bias バイアスパラメータ
     * @param mean 推論時に使用する平均（外部入力）
     * @param inv_variance 推論時に使用する逆分散（外部入力）
     * @param y 出力データポインタ
     * @param workspace ワークスペースバッファ
     * @return ZenuStatus 実行結果
     */
    ZenuStatus forward_inference(const void* x, 
                                 const void* scale, 
                                 const void* bias, 
                                 const void* mean, 
                                 const void* inv_variance, 
                                 void* y,
                                 void* workspace);

    /**
     * @brief 逆伝搬の実行（BN backward）。
     * 順伝搬で保存した mean, inv_variance を用いて入力 x に対する勾配 dx と、
     * scale, bias に対する勾配 dscale, dbias を計算する。
     *
     * @param dy 出力側勾配
     * @param x 順伝搬時の入力
     * @param scale スケールパラメータ
     * @param mean 順伝搬時に得た平均
     * @param inv_variance 順伝搬時に得た逆分散
     * @param dx 入力勾配出力ポインタ
     * @param dscale スケール勾配出力ポインタ
     * @param dbias バイアス勾配出力ポインタ
     * @param workspace ワークスペースバッファ
     * @return ZenuStatus 実行結果
     */
    ZenuStatus backward(const void* dy,
                        const void* x,
                        const void* scale,
                        const void* mean,
                        const void* inv_variance,
                        void* dx,
                        void* dscale,
                        void* dbias,
                        void* workspace);

    /**
     * @brief 順伝搬実行に必要なワークスペースサイズ（バイト数）を取得する。
     */
    size_t get_workspace_bytes_forward();

    /**
     * @brief 逆伝搬実行に必要なワークスペースサイズ（バイト数）を取得する。
     */
    size_t get_workspace_bytes_backward();

private:
    std::vector<size_t> shape_;
    size_t dim_;
    ZenuDataType type_;
    bool is_train_;

    // 順伝搬用グラフ
    std::shared_ptr<fe::graph::Graph> fwd_graph_;
    // 逆伝搬用グラフ（学習時のみ）
    std::shared_ptr<fe::graph::Graph> bwd_graph_;

    // [Forward] 入出力テンソル（名前は識別用）
    std::shared_ptr<fe::graph::Tensor_attributes> X_fwd;
    std::shared_ptr<fe::graph::Tensor_attributes> Y_fwd;
    // オプションでランニング統計を利用する場合のテンソル（ここでは未使用）
    std::shared_ptr<fe::graph::Tensor_attributes> prev_running_mean;
    std::shared_ptr<fe::graph::Tensor_attributes> prev_running_var;
    std::shared_ptr<fe::graph::Tensor_attributes> next_running_mean;
    std::shared_ptr<fe::graph::Tensor_attributes> next_running_var;

    // BN のパラメータ／保存データ用テンソル
    std::shared_ptr<fe::graph::Tensor_attributes> scale_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> bias_tensor;
    std::shared_ptr<fe::graph::Tensor_attributes> mean_tensor;          // forward で算出した平均
    std::shared_ptr<fe::graph::Tensor_attributes> inv_variance_tensor;  // forward で算出した逆分散

    // 定数（epsilon, momentum）
    std::shared_ptr<fe::graph::Tensor_attributes> epsilon;
    std::shared_ptr<fe::graph::Tensor_attributes> momentum;

    // [Backward] 逆伝搬用の入出力テンソル
    std::shared_ptr<fe::graph::Tensor_attributes> dy_bwd;
    std::shared_ptr<fe::graph::Tensor_attributes> dx_bwd;
    std::shared_ptr<fe::graph::Tensor_attributes> dscale_bwd;
    std::shared_ptr<fe::graph::Tensor_attributes> dbias_bwd;

    // 各グラフの初期化関数
    ZenuStatus init_forward_train();
    ZenuStatus init_forward_inference();
    ZenuStatus init_backward();
};

