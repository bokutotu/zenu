#include "zenu_compute_norm.h"
#include "batchnorm.h"

struct ZenuComputeBatchNorm {
    ZenuComputeBatchNormImpl* impl;
};

void zenu_compute_create_batchnorm(ZenuComputeBatchNorm** batchnorm) {
    *batchnorm = new ZenuComputeBatchNorm();
    (*batchnorm)->impl = new ZenuComputeBatchNormImpl();
}

void zenu_compute_destroy_batchnorm(ZenuComputeBatchNorm* batchnorm) {
    delete batchnorm->impl;
    delete batchnorm;
}

ZenuStatus zenu_compute_init_batchnorm(
    ZenuComputeBatchNorm* batchnorm,
    size_t* shape,
    size_t dim,
    ZenuDataType type,
    bool is_train
) {
    return batchnorm->impl->init(std::vector<size_t>(shape, shape + dim), dim, type, is_train);
}

ZenuStatus zenu_compute_forward_batchnorm_inference(
    ZenuComputeBatchNorm* batchnorm,
    const void* input,
    const void* scale,
    const void* bias,
    const void* mean,
    const void* inv_variance,
    void* output,
    void* workspace
) {
    return batchnorm->impl->forward_inference(input, scale, bias, mean, inv_variance, output, workspace);
}

ZenuStatus zenu_compute_forward_batchnorm_train(
    ZenuComputeBatchNorm* batchnorm,
    const void* input,
    const void* scale,
    const void* bias,
    void* output,
    void* mean,
    void* inv_variance,
    void* workspace
) {
    return batchnorm->impl->forward_train(input, scale, bias, output, mean, inv_variance, workspace);
}

ZenuStatus zenu_compute_backward_batchnorm(
    ZenuComputeBatchNorm* batchnorm,
    const void* d_output,
    const void* input,
    const void* scale,
    const void* mean,
    const void* inv_variance,
    void* d_input,
    void* d_scale,
    void* d_bias,
    void* workspace
) {
    return batchnorm->impl->backward(d_output, input, scale, mean, inv_variance, d_input, d_scale, d_bias, workspace);
}

size_t zenu_compute_batchnorm_forward_get_workspace_bytes(
    ZenuComputeBatchNorm* batchnorm
) {
    return batchnorm->impl->get_workspace_bytes_forward();
}

size_t zenu_compute_batchnorm_backward_get_workspace_bytes(
    ZenuComputeBatchNorm* batchnorm
) {
    return batchnorm->impl->get_workspace_bytes_backward();
}
