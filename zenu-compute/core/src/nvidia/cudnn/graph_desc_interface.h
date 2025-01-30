#pragma once

#include <iostream>

#include "cudnn_frontend.h"
#include "zenu_compute_type.h"

namespace fe = cudnn_frontend;

class IGraphDescriptor {
protected:
    fe::graph::Graph graph;
    std::vector<fe::HeurMode_t> heur_mode = {fe::HeurMode_t::A, fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK};

public:
    virtual ~IGraphDescriptor() = default;

    virtual ZenuStatus build_and_check_graph(cudnnHandle_t* handle) {
        auto err = graph.validate();
        if (!err.is_good()) { return handle_error("Graph validation", err); }

        err = graph.build_operation_graph(*handle);
        if (!err.is_good()) { return handle_error("Graph build operation graph", err); }

        err = graph.create_execution_plans(heur_mode);
        if (!err.is_good()) { return handle_error("Graph create execution plans", err); }

        err = graph.check_support(*handle);
        if (!err.is_good()) { return handle_error("Graph check support", err); }

        err = graph.build_plans(*handle, fe::BuildPlanPolicy_t::ALL);
        if (!err.is_good()) { return handle_error("Graph build plans", err); }

        return Success;
    }

    virtual ZenuStatus get_workspace_size(int64_t* workspace_size) {
        auto err = graph.get_workspace_size(*workspace_size);
        if (!err.is_good()) {
            return handle_error("Graph get workspace size", err);
        }
        return Success;
    }

    virtual ZenuStatus execute_graph(cudnnHandle_t* handle, 
                                     std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> &variant_pack,
                                     void* workspace) {
        auto err = graph.execute(*handle, variant_pack, workspace);
        if (!err.is_good()) {
            return handle_error("Graph execute failed", err);
        }
        return Success;
    }

protected:
    ZenuStatus handle_error(const std::string &msg, const fe::error_t &err) {
        std::cout << msg << std::endl;
        std::cout << const_cast<fe::error_t&>(err).get_message() << std::endl;
        return CudnnError;
    }

};

