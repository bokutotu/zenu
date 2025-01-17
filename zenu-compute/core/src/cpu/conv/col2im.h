#include <omp.h>
#include <cstddef>
#include <cstring>

template <typename T>
void col2im2d(
    const T* col,
    int batch_size,
    int channels,
    int height_in,
    int width_in,
    int height_out,
    int width_out,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int workspace_size,
    T* output
)
{
    std::memset(output, 0, sizeof(T) * workspace_size);

    const int col_channels = channels * kernel_h * kernel_w;
    const int col_spatial  = height_out * width_out;

#pragma omp parallel for collapse(2)
    for(int n = 0; n < batch_size; ++n){
        for(int c = 0; c < channels; ++c){
            for(int kh = 0; kh < kernel_h; ++kh){
                for(int kw = 0; kw < kernel_w; ++kw){
                    const int oc = (c * kernel_h + kh) * kernel_w + kw;

                    const T* col_ptr_ck = col
                        + static_cast<size_t>(n) * col_channels * col_spatial
                        + static_cast<size_t>(oc) * col_spatial;

                    T* out_ptr_nc = output
                        + static_cast<size_t>(n) * (channels * height_in * width_in)
                        + static_cast<size_t>(c) * (height_in * width_in);

                    for(int oh = 0; oh < height_out; ++oh){
                        const int ih = oh*stride_h - pad_h + kh*dilation_h;
                        for(int ow = 0; ow < width_out; ++ow){
                            const int iw = ow*stride_w - pad_w + kw*dilation_w;

                            if(ih >= 0 && ih < height_in && iw >= 0 && iw < width_in){
                                out_ptr_nc[ih*width_in + iw] += col_ptr_ck[oh*width_out + ow];
                            }
                        }
                    }
                }
            }
        }
    }
}


