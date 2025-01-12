#pragma once

#include "zenu_compute_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CPU上のメモリをアライメントを考慮して動的に確保する関数
 *
 * @param[out] ptr       確保したメモリの先頭アドレスを格納するためのポインタ（ポインタのアドレスを渡す）
 * @param[in]  num_bytes 確保したいメモリのバイト数
 * @return ZenuStatus    成功した場合は Success、メモリ不足等で失敗した場合は OutOfMemory を返す
 *
 * この関数は CPU(ホスト)上に、AVX-512/AVX2/NEON などを想定したアライメント（64, 32, 16 など）でメモリを確保します。  
 * `aligned_alloc` を使用しており、バイト数 (\p num_bytes) は指定されたアライメント境界に切り上げられます。  
 * 確保に成功すると、\p ptr が指す変数にメモリブロックの先頭アドレスが設定されます。  
 * 使用後は、必ず \ref zenu_compute_free_cpu() を呼び出してメモリを解放してください。
 */
ZenuStatus zenu_compute_malloc_cpu(void** ptr, int num_bytes);

/**
 * @brief CPU上のメモリを解放する関数
 *
 * @param[in] ptr 解放対象のメモリブロックの先頭アドレス
 *
 * \ref zenu_compute_malloc_cpu() で確保したメモリを解放します。  
 * \p ptr が nullptr の場合は何も行いません。  
 * 内部的には `free` を呼び出してメモリを解放します。
 */
void zenu_compute_free_cpu(void* ptr);

/**
 * @brief CPU上のバッファ \p dst に、値 \p value を繰り返し書き込む関数
 *
 * @param[out] dst       書き込み先のCPUメモリ (配列先頭アドレス)
 * @param[in]  value     書き込む値のアドレス (float* or double* など)
 * @param[in]  num_bytes 書き込みたいバイト数
 * @param[in]  type      データ型 (f32 または f64)
 *
 * - \p type == f32 の場合、\p dst は float 配列とみなし、\p value から読み取った単精度浮動小数 (float) を繰り返し代入します。  
 * - \p type == f64 の場合、実装例では \p dst を int 配列とみなし、\p value から読み取った int 値を繰り返し代入しています。(※実装上の注意が必要)
 *
 */
void zenu_compute_set_cpu(void* dst, void* value, int num_bytes, ZenuDataType type);

/**
 * @brief NVIDIA GPU上にメモリを動的に確保する関数
 *
 * @param[out] ptr       確保したデバイスメモリの先頭アドレスを格納するためのポインタ
 * @param[in]  num_bytes 確保したいメモリのバイト数
 * @return ZenuStatus    成功した場合は Success、失敗した場合は OutOfMemory を返す
 *
 * NVIDIA GPU 上に \p num_bytes 分のデバイスメモリを確保します。  
 * 確保に成功すると、\p ptr が指す変数にデバイスメモリの先頭アドレスが設定されます。  
 * 利用が終わったら、\ref zenu_compute_free_nvidia() で解放してください。
 */
ZenuStatus zenu_compute_malloc_nvidia(void** ptr, int num_bytes);

/**
 * @brief NVIDIA GPUメモリ上のバッファ \p dst 全体を、同じく GPUメモリ上にある \p value[0] の値で埋める関数
 *
 * @param[out] dst       書き込み先 (GPUメモリ上のポインタ)
 * @param[in]  value     GPUメモリ上にある単一要素 (float または double) を指すポインタ  
 *                        - \p type が f32 の場合、\p value は 1つの float 値を格納している  
 *                        - \p type が f64 の場合、\p value は 1つの double 値を格納している
 * @param[in]  num_bytes \p dst に書き込む総バイト数 (要素数 * sizeof(float or double))
 * @param[in]  type      データ型 (f32 または f64)
 *
 * @return ZenuStatus
 *  - Success:         正常に完了
 *  - InvalidArgument: \p dst, \p value が nullptr または \p num_bytes <= 0 の場合
 *  - DeviceError:     CUDAカーネルの起動または同期でエラーが発生した場合
 *
 * 本関数は GPU 上のカーネルを起動し、\p dst の全要素に対して \p value[0] の値を書き込みます。  
 * 具体的には、要素数 = (\p num_bytes / sizeof(\p type)) と見なし、  
 * それぞれの要素を \p value[0] (GPUメモリ上の単一の値) で上書きします。
 *
 * @note \p value は配列ではなく、「1つの要素が入った GPU メモリ上のポインタ」です。  
 *       全ての \p dst の要素に対し、その 1要素目の値が繰り返しコピーされます。
 */
ZenuStatus zenu_compute_set_nvidia(void* dst, void* value, int num_bytes, ZenuDataType type);

/**
 * @brief NVIDIA GPU上のメモリを解放する関数
 *
 * @param[in] ptr 解放対象のデバイスメモリ先頭アドレス
 *
 * \ref zenu_compute_malloc_nvidia() で確保した GPUメモリを解放します。  
 * \p ptr が nullptr の場合は何もしません。  
 * 内部的には `cudaFree(ptr)` を呼び出しています。
 */
void zenu_compute_free_nvidia(void* ptr);

/**
 * @brief CPUメモリから GPUメモリへデータをコピーする関数
 *
 * @param[out] dst       GPUメモリ上のポインタ (コピー先)
 * @param[in]  src       CPUメモリ上のポインタ (コピー元)
 * @param[in]  num_bytes コピーするバイト数
 * @return ZenuStatus    Success (成功) または各種エラー (InvalidArgument, DeviceErrorなど)
 *
 * - \p dst が nullptr であったり、\p src が nullptr であったり、\p num_bytes <= 0 の場合は InvalidArgument を返します。  
 * - 内部実装では `cudaMemcpy(dst, src, num_bytes, cudaMemcpyHostToDevice)` を行い、CUDAの返り値に応じてステータスを変換します。
 */
ZenuStatus zenu_compute_cpu_to_nvidia(void* dst, void* src, int num_bytes);

/**
 * @brief GPUメモリから CPUメモリへデータをコピーする関数
 *
 * @param[out] dst       CPUメモリ上のポインタ (コピー先)
 * @param[in]  src       GPUメモリ上のポインタ (コピー元)
 * @param[in]  num_bytes コピーするバイト数
 * @return ZenuStatus    Success (成功) または各種エラー (InvalidArgument, DeviceErrorなど)
 *
 * - \p dst が nullptr であったり、\p src が nullptr であったり、\p num_bytes <= 0 の場合は InvalidArgument を返します。  
 * - 内部実装では `cudaMemcpy(dst, src, num_bytes, cudaMemcpyDeviceToHost)` を行い、CUDAの返り値に応じてステータスを変換します。
 */
ZenuStatus zenu_compute_nvidia_to_cpu(void* dst, void* src, int num_bytes);

#ifdef __cplusplus
}
#endif

