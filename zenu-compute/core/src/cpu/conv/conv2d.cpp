#include "im2col.h"
#include "zenu_compute_blas.h"
#include <cstring>
#include <omp.h>
#include <cstddef>
#include <cmath>

static inline size_t align_up(size_t addr, size_t alignment) {
    return ((addr + alignment - 1) / alignment) * alignment;
}

/**
 * @brief conv2d の im2col + gemm用に必要なワークスペース (float 配列の要素数) を計算する。
 *
 * @param[in] batch_size
 * @param[in] channels_in
 * @param[in] height_in
 * @param[in] width_in
 * @param[in] channels_out
 * @param[in] kernel_h
 * @param[in] kernel_w
 * @param[in] pad_h
 * @param[in] pad_w
 * @param[in] stride_h
 * @param[in] stride_w
 * @param[in] dilation_h
 * @param[in] dilation_w
 *
 * @return 必要な float 要素数 (これに sizeof(float) をかけるとバイト数)
 */
size_t get_workspace_size_conv2d(
    int batch_size,
    int channels_in,
    int height_in,
    int width_in,
    int channels_out,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w
)
{
    // 1. 出力サイズ (H_out, W_out)
    //    Conv2d の一般的な計算
    //       H_out = floor((H_in + 2*pad_h - (dilation_h*(kernel_h-1) + 1)) / stride_h + 1)
    //       W_out = floor((W_in + 2*pad_w - (dilation_w*(kernel_w-1) + 1)) / stride_w + 1)
    //    ここでは割り切れる想定。負やゼロなら 0 とする etc.
    int H_out = (height_in + 2*pad_h - (dilation_h*(kernel_h-1) + 1)) / stride_h + 1;
    int W_out = (width_in  + 2*pad_w - (dilation_w*(kernel_w-1) + 1)) / stride_w + 1;
    if(H_out < 0) H_out = 0;
    if(W_out < 0) W_out = 0;

    // 2. im2col バッファ要素数
    //    shape: [ (C_in*kH*kW), (N * H_out * W_out) ]
    const size_t K   = (size_t)channels_in * kernel_h * kernel_w;
    const size_t N_  = (size_t)batch_size * H_out * W_out;  // im2colの "列" 次元
    const size_t im2col_elems = K * N_;

    // 3. GEMM 出力バッファ要素数
    //    shape: [ C_out, (N * H_out * W_out) ]
    const size_t M   = (size_t)channels_out;  // C_out
    const size_t gemm_elems   = M * N_;

    // 4. アライメントを考慮して im2col 部分の要素数を切り上げ
    //    例: 64バイト (=16 float) 境界に揃える
    const size_t alignment_in_floats = 16;  // 64バイト / 4
    // im2col_elems の部分を切り上げ
    size_t im2col_aligned = align_up_size_t(im2col_elems, alignment_in_floats);

    // 5. 合計 (単位: float 要素数)
    size_t total = im2col_aligned + gemm_elems;

    return total;
}

void conv2d(
    const float* input,
    const float* kernel,
    int batch_size,
    int channels_in,
    int height_in,
    int width_in,
    int height_out,
    int width_out,
    int channels_out,
    int kernel_h,
    int kernel_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int workspace_size,
    float* workspace,
    float* output
)
{
    // 出力サイズなど
    //   im2col: [ (C_in*kH*kW), (N*H_out*W_out) ]
    const int K   = channels_in * kernel_h * kernel_w;    // im2col の行数
    const int N_  = batch_size * height_out * width_out;  // im2col の列数
    const int M   = channels_out;                         // GEMM の行列Aの行数 (=C_out)

    //-----------------------------
    // 1. バッファのレイアウトを決める
    //    まず im2col のための領域 (col_buf) を確保
    //    その後, 一定のアライメントをあけて gemm用領域 (gemm_buf) を確保
    //-----------------------------
    // 必要サイズ (要素数)
    size_t im2col_elems = (size_t)K * N_;   // im2col
    size_t gemm_elems   = (size_t)M * N_;   // GEMM出力 [C_out, N_] = [M, N_]

    // バイト数
    const size_t FLOAT_SIZE = sizeof(float);
    size_t im2col_bytes = im2col_elems * FLOAT_SIZE;
    size_t gemm_bytes   = gemm_elems   * FLOAT_SIZE;

    // 例: 64バイトアライメント
    const size_t alignment_bytes = 64;

    // im2col 用の先頭 (workspace の先頭を使用)
    float* im2col_buf = workspace;

    // im2col_buf の次に、アライメントを考慮して gemm_buf を配置
    {
        // im2col_buf の末尾バイト = (im2col_buf 先頭アドレス + im2col_bytes)
        // ただし float* → byte のオフセットを計算するために、まずポインタ→uintptr_tに変換
        uintptr_t base_addr = reinterpret_cast<uintptr_t>(workspace);
        uintptr_t im2col_end = base_addr + im2col_bytes;

        // ここでアライメントをかけて切り上げ
        uintptr_t gemm_start = align_up(im2col_end, alignment_bytes);

        // gemm_buf の float* ポインタに変換
        float* gemm_buf = reinterpret_cast<float*>(gemm_start);

        // 以降のGEMM出力先として使う
        // (実際にバッファ量が足りるかどうか workspace_size と照合しておくのが望ましい)
        size_t used_for_gemm = gemm_elems;  // 要素数
        size_t gemm_buf_end_elems = (gemm_start - base_addr)/FLOAT_SIZE + used_for_gemm; // 全体終了位置(要素単位)

        // ざっくりチェック: gemm_buf_end_elems <= workspace_size
        if(gemm_buf_end_elems > (size_t)workspace_size){
            // メモリ不足などのエラー処理
            // ここでは assert か何か
            fprintf(stderr, "[conv2d] workspace too small!\n");
            return;
        }

        //--------------------------------
        // 2. im2col
        //--------------------------------
        im2col2d(
            input,
            batch_size,
            channels_in,
            height_in,
            width_in,
            height_out,
            width_out,
            kernel_h,
            kernel_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            im2col_buf
        );

        //--------------------------------
        // 3. GEMM呼び出し
        //--------------------------------
        // A = kernel ([C_out, K]) -> row-major => Aのshape=(M,K)
        // B = im2col_buf ([K,N_]) -> Bのshape=(K,N_)
        // C = gemm_buf ([M,N_])
        const int lda = K;
        const int ldb = N_;
        const int ldc = N_;

        const double alpha = 1.0;
        const double beta  = 0.0;

        // ここで単精度 (f32) 用に SGEMM を呼び出し
        zenu_compute_gemm_cpu(
            NoTranspose,   // transA
            NoTranspose,   // transB
            M,                   // 行列Cの行数
            N_,                  // 行列Cの列数
            K,                   // op(A)の列数=op(B)の行数
            alpha,
            kernel,              // A
            lda,
            im2col_buf,          // B
            ldb,
            beta,
            gemm_buf,            // C
            ldc,
            f32
        );

        //--------------------------------
        // 4. reshape: gemm_buf -> output
        //--------------------------------
#pragma omp parallel for
        for(int n = 0; n < batch_size; ++n){
            for(int oc = 0; oc < channels_out; ++oc){
                for(int oh = 0; oh < height_out; ++oh){
                    for(int ow = 0; ow < width_out; ++ow){
                        int col_index = n*(height_out*width_out) + oh*width_out + ow;
                        int gemm_index = oc*N_ + col_index;
                        int out_index = n*(channels_out*height_out*width_out)
                                      + oc*(height_out*width_out)
                                      + oh*(width_out)
                                      + ow;
                        output[out_index] = gemm_buf[gemm_index];
                    }
                }
            }
        }
    }
}

