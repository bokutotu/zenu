#include "conv_interface.h"
#include "nvidia/cudnn/utils.h"
#include "nvidia/handle.h"

#include <cudnn_frontend.h>
#include <memory>

namespace fe = cudnn_frontend;

ZenuStatus ZenuComputeConvNvidiaImpl::init(std::vector<size_t> input, 
                                std::vector<size_t> output, 
                                std::vector<size_t> kernel, 
                                std::vector<size_t> stride, 
                                std::vector<size_t> padding, 
                                std::vector<size_t> dilation,
                                ZenuDataType type) {
    fwd_graph = std::make_shared<fe::graph::Graph>();
    bwd_data_graph = std::make_shared<fe::graph::Graph>();
    bwd_kernel_graph = std::make_shared<fe::graph::Graph>();

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

    auto st = init_fwd();
    if (st != Success) {
        return st;
    }
    // st = init_bwd_data();
    // if (st != Success) {
    //     return st;
    // }
    // st = init_bwd_kernel();
    return st;
}

ZenuStatus ZenuComputeConvNvidiaImpl::init_fwd() {
    fwd_graph->set_io_data_type(get_data_type(type))
             .set_compute_data_type(get_data_type(type));

    X_fwd = fwd_graph->tensor(get_tensor_attributes(input, type));
    Kernel_fwd = fwd_graph->tensor(get_tensor_attributes(kernel, type));

    conv_options = fe::graph::Conv_fprop_attributes()
                        .set_padding(convert(padding))
                        .set_stride(convert(stride))
                        .set_dilation(convert(dilation));

    Y_fwd = fwd_graph->conv_fprop(X_fwd, Kernel_fwd, conv_options);
    Y_fwd->set_output(true)
          .set_dim(convert(output))
          .set_stride(default_stride(output));

    std::vector<fe::HeurMode_t> heur_modes = {fe::HeurMode_t::A, fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK};
    auto st = build_and_check_graph(*fwd_graph, heur_modes);

    return st;
}

size_t ZenuComputeConvNvidiaImpl::get_forward_bytes() const {
    return get_workspace_size(*fwd_graph);
}

ZenuStatus ZenuComputeConvNvidiaImpl::forward(const void* input, 
                                              const void* kernel, 
                                              void* output, 
                                              void* workspace) const {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X_fwd, const_cast<void*>(input)},
        {Kernel_fwd, const_cast<void*>(kernel)},
        {Y_fwd, output}
    };

    auto st = fwd_graph->execute(NvidiaHandles::getCudnnHandle(), variant_pack, workspace);
    if (st.is_good()) {
        return Success;
    } else {
        return CudnnError;
    }
}

