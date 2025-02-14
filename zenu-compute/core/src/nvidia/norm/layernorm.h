#pragma once

#include "base.h"
#include "nvidia/cudnn/utils.h"
#include "zenu_compute_type.h"

class ZenuComputeLayerNormNvidiaImpl : public ZenuComputeNormNvidiaImpl {
public:
    ~ZenuComputeLayerNormNvidiaImpl() override = default;

    ZenuStatus init(const std::vector<size_t>& input_shape,
                    const std::vector<size_t>& param_shape,
                    bool training_mode,
                    NormMode mode,
                    ZenuDataType type) override {
        // 保存
        input_shape_ = input_shape;
        param_shape_ = param_shape;
        training_mode_ = training_mode;
        mode_ = mode;
        type_ = type;

        // --- Forward グラフ構築 ---
        fwd_graph_ = std::make_shared<fe::graph::Graph>();
        fwd_graph_->set_io_data_type(get_data_type(type_))
                  .set_compute_data_type(get_data_type(type_));

        X_ln_fwd_ = fwd_graph_->tensor(get_tensor_attributes(input_shape_, type_));

        // scale, bias テンソル（通常は param_shape_ と同じ）
        auto p_stride = compute_default_stride(param_shape_);
        scale_ln_fwd_ = fwd_graph_->tensor(get_tensor_attributes(param_shape_, type_));
        bias_ln_fwd_ = fwd_graph_->tensor(get_tensor_attributes(param_shape_, type_));

        // epsilon テンソル
        float epsilon_val = 1e-5f;
        auto epsilon_tensor = fwd_graph_->tensor(epsilon_val);

        auto ln_options = fe::graph::Layernorm_attributes()
                              .set_epsilon(epsilon_tensor)
                              .set_forward_phase(training_mode_ ? fe::NormFwdPhase_t::TRAINING : fe::NormFwdPhase_t::INFERENCE);

        if (training_mode_) {
            auto ln_ret = fwd_graph_->layernorm(X_ln_fwd_, scale_ln_fwd_, bias_ln_fwd_, ln_options);
            Y_ln_fwd_ = std::get<0>(ln_ret);
            mean_ln_ = std::get<1>(ln_ret);
            inv_variance_ln_ = std::get<2>(ln_ret);
            Y_ln_fwd_->set_output(true);
            mean_ln_->set_output(true).set_data_type(fe::DataType_t::FLOAT);
            inv_variance_ln_->set_output(true).set_data_type(fe::DataType_t::FLOAT);
        }
        else {
            // 推論時：mean, inv_variance は生成されない
            Y_ln_fwd_ = fwd_graph_->layernorm(X_ln_fwd_, scale_ln_fwd_, bias_ln_fwd_, ln_options);
            Y_ln_fwd_->set_output(true);
            mean_ln_ = nullptr;
            inv_variance_ln_ = nullptr;
        }

        // Forward グラフ構築
        {
            auto handle_ptr = create_cudnn_handle();
            auto handle = *handle_ptr;
            if (!fwd_graph_->validate().is_good()) return ZenuStatus::FAIL;
            if (!fwd_graph_->build_operation_graph(handle).is_good()) return ZenuStatus::FAIL;
            if (!fwd_graph_->create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good()) return ZenuStatus::FAIL;
            if (!fwd_graph_->check_support(handle).is_good()) return ZenuStatus::FAIL;
            if (!fwd_graph_->build_plans(handle).is_good()) return ZenuStatus::FAIL;
        }

        // --- Backward グラフ構築 ---
        bwd_graph_ = std::make_shared<fe::graph::Graph>();
        if (type_ == ZenuDataType::FP32)
        {
            bwd_graph_->set_io_data_type(fe::DataType_t::FLOAT)
                       .set_intermediate_data_type(fe::DataType_t::FLOAT)
                       .set_compute_data_type(fe::DataType_t::FLOAT);
        }
        else if (type_ == ZenuDataType::FP16)
        {
            bwd_graph_->set_io_data_type(fe::DataType_t::HALF)
                       .set_intermediate_data_type(fe::DataType_t::FLOAT)
                       .set_compute_data_type(fe::DataType_t::FLOAT);
        }

        // Backward 用入力テンソル X
        X_ln_bwd_ = bwd_graph_->tensor(fe::graph::Tensor_attributes()
                          .set_name("X")
                          .set_dim(input_shape_)
                          .set_stride(x_stride));
        // scale テンソル（backward 用）
        scale_ln_bwd_ = bwd_graph_->tensor(fe::graph::Tensor_attributes()
                          .set_name("scale")
                          .set_dim(param_shape_)
                          .set_stride(p_stride)
                          .set_data_type(fe::DataType_t::FLOAT));
        // 出力勾配 dY
        dY_ln_ = bwd_graph_->tensor(fe::graph::Tensor_attributes()
                          .set_name("dY")
                          .set_dim(input_shape_)
                          .set_stride(x_stride));

        // LayerNorm backward オプションの設定：学習時は saved mean/inv_variance を入力
        auto ln_bwd_options = fe::graph::Layernorm_backward_attributes();
        if (training_mode_) {
            ln_bwd_options.set_saved_mean_and_inv_variance(mean_ln_, inv_variance_ln_);
        }
        // backward の呼び出し：戻り値は (dX, dscale, dbias)
        auto ln_bwd_ret = bwd_graph_->layernorm_backward(dY_ln_, X_ln_bwd_, scale_ln_bwd_, ln_bwd_options);
        grad_X_ln_ = std::get<0>(ln_bwd_ret);
        grad_scale_ln_ = std::get<1>(ln_bwd_ret);
        grad_bias_ln_  = std::get<2>(ln_bwd_ret);
        grad_X_ln_->set_output(true);
        grad_scale_ln_->set_output(true).set_data_type(fe::DataType_t::FLOAT);
        grad_bias_ln_->set_output(true).set_data_type(fe::DataType_t::FLOAT);

        // Backward グラフ構築
        {
            auto handle_ptr = create_cudnn_handle();
            auto handle = *handle_ptr;
            if (!bwd_graph_->validate().is_good()) return ZenuStatus::FAIL;
            if (!bwd_graph_->build_operation_graph(handle).is_good()) return ZenuStatus::FAIL;
            if (!bwd_graph_->create_execution_plans({fe::HeurMode_t::FALLBACK}).is_good()) return ZenuStatus::FAIL;
            if (!bwd_graph_->check_support(handle).is_good()) return ZenuStatus::FAIL;
            if (!bwd_graph_->build_plans(handle).is_good()) return ZenuStatus::FAIL;
        }

        return ZenuStatus::Success;
    }

    size_t get_forward_bytes() const override {
        size_t bytes = 0;
        if (fwd_graph_) fwd_graph_->get_workspace_size(bytes);
        return bytes;
    }
    size_t get_backward_bytes() const override {
        size_t bytes = 0;
        if (bwd_graph_) bwd_graph_->get_workspace_size(bytes);
        return bytes;
    }

    ZenuStatus forward(const void* input,
                       void* output,
                       const void* scale,
                       const void* bias,
                       void* saved_mean,
                       void* saved_inv_variance,
                       void* workspace) const override {
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
        variant_pack[X_ln_fwd_] = const_cast<void*>(input);
        variant_pack[scale_ln_fwd_] = const_cast<void*>(scale);
        variant_pack[bias_ln_fwd_]  = const_cast<void*>(bias);
        variant_pack[Y_ln_fwd_]     = output;
        if (training_mode_) {
            variant_pack[mean_ln_] = saved_mean;
            variant_pack[inv_variance_ln_] = saved_inv_variance;
        }
        auto handle_ptr = create_cudnn_handle();
        auto handle = *handle_ptr;
        return fwd_graph_->execute(handle, variant_pack, workspace);
    }

    ZenuStatus backward(const void* grad_output,
                        const void* input,
                        const void* scale,
                        const void* saved_mean,
                        const void* saved_inv_variance,
                        void* grad_input,
                        void* grad_scale,
                        void* grad_bias,
                        void* workspace) const override {
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack;
        variant_pack[X_ln_bwd_] = const_cast<void*>(input);
        variant_pack[scale_ln_bwd_] = const_cast<void*>(scale);
        variant_pack[dY_ln_] = const_cast<void*>(grad_output);
        variant_pack[grad_X_ln_] = grad_input;
        variant_pack[grad_scale_ln_] = grad_scale;
        variant_pack[grad_bias_ln_]  = grad_bias;
        if (training_mode_) {
            variant_pack[mean_ln_] = const_cast<void*>(saved_mean);
            variant_pack[inv_variance_ln_] = const_cast<void*>(saved_inv_variance);
        }
        auto handle_ptr = create_cudnn_handle();
        auto handle = *handle_ptr;
        return bwd_graph_->execute(handle, variant_pack, workspace);
    }

private:
    // forward 用 tensor 属性
    std::shared_ptr<fe::graph::Tensor_attributes> X_ln_fwd_;
    std::shared_ptr<fe::graph::Tensor_attributes> scale_ln_fwd_;
    std::shared_ptr<fe::graph::Tensor_attributes> bias_ln_fwd_;
    std::shared_ptr<fe::graph::Tensor_attributes> Y_ln_fwd_;
    // 学習時のみ生成される統計量
    std::shared_ptr<fe::graph::Tensor_attributes> mean_ln_;
    std::shared_ptr<fe::graph::Tensor_attributes> inv_variance_ln_;

    // backward 用 tensor 属性
    std::shared_ptr<fe::graph::Tensor_attributes> X_ln_bwd_;
    std::shared_ptr<fe::graph::Tensor_attributes> scale_ln_bwd_;
    std::shared_ptr<fe::graph::Tensor_attributes> dY_ln_;
    std::shared_ptr<fe::graph::Tensor_attributes> grad_X_ln_;
    std::shared_ptr<fe::graph::Tensor_attributes> grad_scale_ln_;
    std::shared_ptr<fe::graph::Tensor_attributes> grad_bias_ln_;
};

