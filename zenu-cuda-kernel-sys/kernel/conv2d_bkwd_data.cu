/******************************************************************************
 * Conv2D Bias Backward 
 *
 *   dOut: [N, C, H, W] の出力勾配 (NCHW)
 *   dbias: [C] の bias 勾配
 *
 *   このコードでは
 *     dbias[c] = sum_{n,h,w}( dOut[n,c,h,w] )
 *   を求める
 *
 * テンプレート T で float / double を切り替え、
 * 将来的に BF16 や TF32 用に拡張可能な設計を意図。
 *
 * Warp Shuffle を使った高速 reduce を実装し、
 * ブロックごとに partial sum を集約、atomicAdd で dbias[c] に加算。
 ******************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

//==============================================================================
// 1. GPU アーキテクチャ互換性対応の atomicAdd 実装
//    (float 版は標準、double 版は __CUDA_ARCH__ >= 600 ならネイティブ、
//     それ未満なら一応 fallback 実装)
//==============================================================================

__device__ inline void atomicAddFloat(float* address, float val) {
    atomicAdd(address, val);
}

// double 版の atomicAdd
#if __CUDA_ARCH__ >= 600
// SM 6.0 (Pascal) 以降ならネイティブの double atomicAdd が利用可能
__device__ inline void atomicAddDouble(double* address, double val) {
    atomicAdd(address, val);
}
#else
// それより古いアーキテクチャの場合、Fallback 実装（性能は低い）
__device__ inline void atomicAddDouble(double* address, double val) {
    unsigned long long int* ptr = (unsigned long long int*) address;
    unsigned long long int old, assumed;
    do {
        old = *ptr;
        assumed = old;
        double f = __longlong_as_double(old);
        f += val;
        old = __double_as_longlong(f);
    } while (atomicCAS(ptr, assumed, old) != assumed);
}
#endif

// テンプレート atomicAdd: T が float の場合と double の場合を切り替え
template <typename T>
__device__ inline void atomicAddT(T* address, T val);

template <>
__device__ inline void atomicAddT<float>(float* address, float val) {
    atomicAddFloat(address, val);
}

template <>
__device__ inline void atomicAddT<double>(double* address, double val) {
    atomicAddDouble(address, val);
}


//==============================================================================
// 2. Warp Shuffle による高速リダクション (warpReduceSum)
//    1 warp (最大 32 スレッド) 内での総和
//==============================================================================

template <typename T>
__device__ inline T warpReduceSum(T val) {
    // ここでは 32 スレッドの warp を仮定
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

//==============================================================================
// 3. Block 内での総和をとる (ブロックサイズ最大 1024 程度)
//    まずは各スレッドで warpReduceSum し、ブロック内の各 warp の結果を
//    さらに 1 warp で纏める
//==============================================================================

template <typename T>
__device__ inline T blockReduceSum(T val) {
    static __shared__ T shared[32];  // 各 warp の部分和を格納

    int warpId = threadIdx.x >> 5; // /32
    val = warpReduceSum(val);

    // warp の代表スレッド (lane == 0) が共有メモリに書き込む
    if ((threadIdx.x & 31) == 0) {
        shared[warpId] = val;
    }
    __syncthreads();

    // ブロック内の warp 数
    int warpCount = blockDim.x >> 5;  // /32
    if (warpCount == 0) warpCount = 1;

    // 先頭 warp だけで最終的な総和を計算
    T result = (threadIdx.x < 32) ? shared[threadIdx.x] : (T)0;
    if (warpId == 0) {
        result = warpReduceSum(result);
    }
    return result;
}

//==============================================================================
// 4. Conv2D bias bwd カーネル
//    dOut: (N, C, H, W) => shape: N*C*H*W
//    dbias: (C)
//==============================================================================
template <typename T>
__global__
void conv2d_bias_backward_kernel(
    const T* __restrict__ dOut,
    T* __restrict__ dbias,
    int N, int C, int H, int W)
{
    int c = blockIdx.x;  // グリッド x 次元がチャネル数
    if (c >= C) return;

    // 各チャネルの先頭オフセット (NCHW レイアウト)
    // N だけが最外なので N*H*W がチャネル c の要素数
    // メモリは [c * (N*H*W)] から開始とする
    const T* dOut_c = dOut + static_cast<long long>(c) * (N * H * W);

    // まず各スレッドが担当する領域の総和をとる
    // blockDim.x threads で N*H*W をループ
    T threadSum = (T)0;
    int total = N * H * W;
    // int tid = threadIdx.x + blockDim.x * blockIdx.y; // blockIdx.y は 1 かもしれないが拡張想定
    // int stride = blockDim.x * gridDim.y;            // 現在は blockIdx.y=1 前提なら stride=blockDim.x

    // 大きな配列を飛び飛びにアクセス (tid, tid+stride, tid+2*stride, ...)
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        threadSum += dOut_c[idx];
    }

    // Block 内で warp shuffle reduce
    T blockSum = blockReduceSum(threadSum);

    // ブロック内スレッド 0 が dbias[c] に加算
    if (threadIdx.x == 0) {
        atomicAddT(&dbias[c], blockSum);
    }
}

//==============================================================================
// 5. エントリーポイント: Conv2D bias backward 呼び出し
//    (float / double) テンプレート
//==============================================================================
template <typename T>
void conv2d_bias_backward(
    const T* dOut,  // device ptr
    T* dbias,       // device ptr
    int N, int C, int H, int W,
    cudaStream_t stream = 0,
    int blockSize = 256)
{
    // dbias を 0 クリア
    cudaMemsetAsync(dbias, 0, C * sizeof(T), stream);

    // グリッドは C 個のブロック (x 次元)
    // y 次元は通常 1 で十分。将来的な拡張で 2D グリッドを使うなら blockIdx.y も考慮
    dim3 grid(C, 1, 1);
    dim3 block(blockSize);

    conv2d_bias_backward_kernel<T><<<grid, block, 0, stream>>>(dOut, dbias, N, C, H, W);
}

void conv2d_bias_bkwd_float(
    const float* dOut,  // device ptr
    float* dbias,       // device ptr
    int N, int C, int H, int W
    )
{
    conv2d_bias_backward<float>(dOut, dbias, N, C, H, W, 0, 256);
}

void conv2d_bias_bkwd_double(
    const double* dOut,  // device ptr
    double* dbias,       // device ptr
    int N, int C, int H, int W)
{
    conv2d_bias_backward<double>(dOut, dbias, N, C, H, W, 0, 256);
}
