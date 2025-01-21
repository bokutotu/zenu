#pragma once

#include "zenu_compute_type.h"

#include <cstddef>
#include <vector>

struct ZenuComputeConvCpu {
public:
    ZenuComputeConvCpu();
    ~ZenuComputeConvCpu();

    ZenuStatus init(std::vector<size_t> input, 
                    std::vector<size_t> output, 
                    std::vector<size_t> kernel, 
                    std::vector<size_t> stride, 
                    std::vector<size_t> padding, 
                    std::vector<size_t> dilation,
                    ZenuDataType type);

    size_t get_forward_output_bytes() const;
    size_t get_backward_data_bytes() const;
    size_t get_backward_kernel_bytes() const;

    ZenuStatus forward(const void* input, const void* kernel, void* output) const;
    ZenuStatus backward_data(const void* kernel, const void* grad_output, void* grad_input) const;
    ZenuStatus backward_kernel(const void* input, const void* grad_output, void* grad_kernel) const;
private:
    std::vector<size_t> input;
    std::vector<size_t> output;
    std::vector<size_t> kernel;
    std::vector<size_t> stride;
    std::vector<size_t> padding;
    std::vector<size_t> dilation;
    ZenuDataType type;

    size_t get_dim() const ;

    size_t get_im2col_bytes() const ;
    size_t get_im2col2d_bytes() const ;

    size_t get_col2im_bytes() const;
    size_t get_col2im2d_bytes() const;

    size_t get_gemm_bytes() const;

    void im2col(const void* input, void* col) const;
    void im2col2d(const void* input, void* col)const;

    void col2im(const void* col, void* input) const;
    void col2im2d(const void* col, void* input) const;

    void transpose_gemm(const void* gemm_out, void* output) const;
    void transpose_gemm2d(const void* gemm_out, void* output) const;
};
