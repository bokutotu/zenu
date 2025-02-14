#include "batchnorm.h"
#include "nvidia/handle.h"
#include "nvidia/cudnn/utils.h"

#include <unordered_map>

//
// 初期化：グラフの作成（training 時は順伝搬＋逆伝搬、inference 時は順伝搬のみ）
//
ZenuStatus ZenuComputeBatchNormImpl::init(std::vector<size_t> shape, size_t dim, ZenuDataType type, bool is_train) {
    shape_ = shape;
    dim_   = dim;
    type_  = type;
    is_train_ = is_train;

    ZenuStatus st = Success;
    if (is_train_) {
        st = init_forward_train();
        if (st != Success) return st;
        st = init_backward();
        if (st != Success) return st;
    } else {
        st = init_forward_inference();
        if (st != Success) return st;
    }
    return st;
}

//
// 順伝搬（training 用）のグラフ初期化
//
ZenuStatus ZenuComputeBatchNormImpl::init_forward_train() {
    fwd_graph_ = std::make_shared<fe::graph::Graph>();
    fwd_graph_->set_io_data_type(get_data_type(type_))
               .set_intermediate_data_type(fe::DataType_t::FLOAT)
               .set_compute_data_type(fe::DataType_t::FLOAT);

    // 入力テンソル X
    X_fwd = fwd_graph_->tensor(
        get_tensor_attributes(shape_, type_)
    );

    // パラメータ用テンソルの shape（通常 {1, C, 1, 1} となる）
    std::vector<size_t> param_shape { 1, shape_[1], 1, 1 };
    // ストライドはユーティリティ関数 default_stride を利用
    auto param_stride = default_stride(param_shape);

    scale_tensor = fwd_graph_->tensor(
        get_tensor_attributes(param_shape, type_)
    );
    scale_tensor->set_stride(param_stride);

    bias_tensor = fwd_graph_->tensor(
        get_tensor_attributes(param_shape, type_)
    );
    bias_tensor->set_stride(param_stride);

    // 順伝搬時は、内部で算出される mean, inv_variance も出力とする
    mean_tensor = fwd_graph_->tensor(
        get_tensor_attributes(param_shape, type_)
    );
    mean_tensor->set_stride(param_stride);

    inv_variance_tensor = fwd_graph_->tensor(
        get_tensor_attributes(param_shape, type_)
    );
    inv_variance_tensor->set_stride(param_stride);

    // 定数（epsilon, momentum）など
    epsilon = std::make_shared<fe::graph::Tensor_attributes>(1e-05f);
    momentum = std::make_shared<fe::graph::Tensor_attributes>(1e-01f);

    // BN 用オプションを構築（running statistics の更新はここでは行わない例）
    auto bn_options = fe::graph::Batchnorm_attributes().set_epsilon(epsilon);
    // bn_options.set_previous_running_stats(...);  // 必要に応じて

    // BatchNorm 順伝搬（training）グラフ作成
    // 戻り値はタプル： (出力, mean, inv_variance)
    auto bn_outputs = fwd_graph_->batchnorm(X_fwd, scale_tensor, bias_tensor, bn_options);
    Y_fwd = std::get<0>(bn_outputs);
    mean_tensor = std::get<1>(bn_outputs);
    inv_variance_tensor = std::get<2>(bn_outputs);

    Y_fwd->set_output(true);
    mean_tensor->set_output(true);
    inv_variance_tensor->set_output(true);

    // グラフのビルドと検証（heuristic モードは FALLBACK など）
    std::vector<fe::HeurMode_t> heur_modes = { fe::HeurMode_t::FALLBACK };
    return build_and_check_graph(*fwd_graph_, heur_modes);
}

//
// 順伝搬（inference 用）のグラフ初期化
//
ZenuStatus ZenuComputeBatchNormImpl::init_forward_inference() {
    fwd_graph_ = std::make_shared<fe::graph::Graph>();
    fwd_graph_->set_io_data_type(get_data_type(type_))
               .set_intermediate_data_type(fe::DataType_t::FLOAT)
               .set_compute_data_type(fe::DataType_t::FLOAT);

    X_fwd = fwd_graph_->tensor(
        get_tensor_attributes(shape_, type_)
    );

    std::vector<size_t> param_shape = { 1, shape_[1], 1, 1 };
    auto param_stride = default_stride(param_shape);

    scale_tensor = fwd_graph_->tensor(
        get_tensor_attributes(param_shape, type_)
    );
    scale_tensor->set_stride(param_stride);

    bias_tensor = fwd_graph_->tensor(
        get_tensor_attributes(param_shape, type_)
    );
    bias_tensor->set_stride(param_stride);

    // 推論時は mean, inv_variance は外部入力となる
    mean_tensor = fwd_graph_->tensor(
        get_tensor_attributes(param_shape, type_)
    );
    mean_tensor->set_stride(param_stride);

    inv_variance_tensor = fwd_graph_->tensor(
        get_tensor_attributes(param_shape, type_)
    );
    inv_variance_tensor->set_stride(param_stride);

    auto bn_infer_options = fe::graph::Batchnorm_inference_attributes();
    Y_fwd = fwd_graph_->batchnorm_inference(X_fwd, mean_tensor, inv_variance_tensor, scale_tensor, bias_tensor, bn_infer_options);
    Y_fwd->set_output(true);

    std::vector<fe::HeurMode_t> heur_modes = { fe::HeurMode_t::FALLBACK };
    return build_and_check_graph(*fwd_graph_, heur_modes);
}

//
// 逆伝搬グラフの初期化（training 時のみ）
//
ZenuStatus ZenuComputeBatchNormImpl::init_backward() {
    bwd_graph_ = std::make_shared<fe::graph::Graph>();
    bwd_graph_->set_io_data_type(get_data_type(type_))
               .set_intermediate_data_type(fe::DataType_t::FLOAT)
               .set_compute_data_type(fe::DataType_t::FLOAT);

    // 逆伝搬用入力：dy
    dy_bwd = bwd_graph_->tensor(
        get_tensor_attributes(shape_, type_)
    );

    // 順伝搬時の入力 x を再利用（逆伝搬用に別オブジェクトとして作成）
    auto X_for_bwd = bwd_graph_->tensor(
        get_tensor_attributes(shape_, type_)
    );

    // BN backward オプション：順伝搬で算出された mean, inv_variance を指定
    auto bn_bwd_options = fe::graph::Batchnorm_backward_attributes()
                              .set_saved_mean_and_inv_variance(mean_tensor, inv_variance_tensor);

    // 逆伝搬グラフ作成。戻り値はタプル： (dx, dscale, dbias)
    auto bn_bwd_outputs = bwd_graph_->batchnorm_backward(dy_bwd, X_for_bwd, scale_tensor, bn_bwd_options);
    dx_bwd = std::get<0>(bn_bwd_outputs);
    dscale_bwd = std::get<1>(bn_bwd_outputs);
    dbias_bwd = std::get<2>(bn_bwd_outputs);

    dx_bwd->set_output(true);
    dscale_bwd->set_output(true);
    dbias_bwd->set_output(true);

    std::vector<fe::HeurMode_t> heur_modes = { fe::HeurMode_t::FALLBACK };
    return build_and_check_graph(*bwd_graph_, heur_modes);
}

//
// 学習時順伝搬実行
//
ZenuStatus ZenuComputeBatchNormImpl::forward_train(const void* x, 
                                                    const void* scale, 
                                                    const void* bias, 
                                                    void* y, 
                                                    void* mean, 
                                                    void* inv_variance,
                                                    void* workspace) {
    // variant_pack の構築：各テンソル属性と実データポインタの対応付け
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        { X_fwd, const_cast<void*>(x) },
        { scale_tensor, const_cast<void*>(scale) },
        { bias_tensor, const_cast<void*>(bias) },
        { Y_fwd, y },
        { this->mean_tensor, mean },
        { this->inv_variance_tensor, inv_variance }
    };

    auto handle = NvidiaHandles::getCudnnHandle();
    auto st = fwd_graph_->execute(handle, variant_pack, workspace);
    return st.is_good() ? Success : CudnnError;
}

//
// 推論時順伝搬実行
//
ZenuStatus ZenuComputeBatchNormImpl::forward_inference(const void* x, 
                                                        const void* scale, 
                                                        const void* bias, 
                                                        const void* mean, 
                                                        const void* inv_variance, 
                                                        void* y,
                                                        void* workspace) {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        { X_fwd, const_cast<void*>(x) },
        { scale_tensor, const_cast<void*>(scale) },
        { bias_tensor, const_cast<void*>(bias) },
        { this->mean_tensor, const_cast<void*>(mean) },
        { this->inv_variance_tensor, const_cast<void*>(inv_variance) },
        { Y_fwd, y }
    };

    auto handle = NvidiaHandles::getCudnnHandle();
    auto st = fwd_graph_->execute(handle, variant_pack, workspace);
    return st.is_good() ? Success : CudnnError;
}

//
// 逆伝搬実行
//
ZenuStatus ZenuComputeBatchNormImpl::backward(const void* dy,
                                               const void* x,
                                               const void* scale,
                                               const void* mean,
                                               const void* inv_variance,
                                               void* dx,
                                               void* dscale,
                                               void* dbias,
                                               void* workspace) {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        { dy_bwd, const_cast<void*>(dy) },
        { X_fwd, const_cast<void*>(x) },
        { scale_tensor, const_cast<void*>(scale) },
        { this->mean_tensor, const_cast<void*>(mean) },
        { this->inv_variance_tensor, const_cast<void*>(inv_variance) },
        { dx_bwd, dx },
        { dscale_bwd, dscale },
        { dbias_bwd, dbias }
    };

    auto handle = NvidiaHandles::getCudnnHandle();
    auto st = bwd_graph_->execute(handle, variant_pack, workspace);
    return st.is_good() ? Success : CudnnError;
}

//
// 順伝搬用ワークスペースサイズ取得
//
size_t ZenuComputeBatchNormImpl::get_workspace_bytes_forward() {
    return get_workspace_size(*fwd_graph_);
}

//
// 逆伝搬用ワークスペースサイズ取得
//
size_t ZenuComputeBatchNormImpl::get_workspace_bytes_backward() {
    return get_workspace_size(*bwd_graph_);
}
