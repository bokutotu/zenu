#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "zenu_compute_type.h"

/**
 * @brief CPU上で正規分布乱数を生成し、dst に書き込む
 *
 * @param dst          出力バッファ (float* or double*)
 * @param num_elements 要素数
 * @param mean         平均
 * @param stddev       標準偏差 (>=0)
 * @param data_type    f32 or f64
 * @param seed         乱数シード (64bit)
 * @return ZenuStatus  成功 or 各種エラー
 */
ZenuStatus zenu_compute_normal_distribution_cpu(
    void* dst,
    int   num_elements,
    float mean,
    float stddev,
    ZenuDataType data_type,
    unsigned long long seed
);

/**
 * @brief CPU上で一様分布乱数を生成し、dst に書き込む
 *
 * @param dst          出力バッファ (float* or double*)
 * @param num_elements 要素数
 * @param low          一様分布の下限
 * @param high         一様分布の上限
 * @param data_type    f32 or f64
 * @param seed         乱数シード (64bit)
 * @return ZenuStatus  成功 or 各種エラー
 */
ZenuStatus zenu_compute_uniform_distribution_cpu(
    void* dst,
    int   num_elements,
    float low,
    float high,
    ZenuDataType data_type,
    unsigned long long seed
);

/**
 * @brief NVIDIA GPU上で正規分布乱数を生成し、dst(GPU)に書き込む。
 *        シードも外部から指定可能。
 *
 * @param dst          GPUメモリ上のポインタ (float* or double*)
 * @param num_elements 生成する乱数の個数
 * @param mean         平均
 * @param stddev       標準偏差 (>=0)
 * @param data_type    f32 or f64
 * @param device_id    使用するGPU ID
 * @param seed         乱数シード (64bit)
 * @return ZenuStatus  成功 or エラー
 */
ZenuStatus zenu_compute_normal_distribution_nvidia(
    void* dst,
    int  num_elements,
    float mean,
    float stddev,
    ZenuDataType data_type,
    int device_id,
    unsigned long long seed
);

/**
 * @brief NVIDIA GPU上で一様分布乱数を生成し、dst(GPU)に書き込む
 *        シードも外部から指定。
 *
 * @param dst          GPUメモリ上のポインタ (float* or double*)
 * @param num_elements 生成する乱数の個数
 * @param low          一様分布の下限
 * @param high         一様分布の上限
 * @param data_type    f32 or f64
 * @param device_id    GPU ID
 * @param seed         乱数シード (64bit)
 * @return ZenuStatus  成功 or エラー
 */
ZenuStatus zenu_compute_uniform_distribution_nvidia(
    void* dst,
    int  num_elements,
    float low,
    float high,
    ZenuDataType data_type,
    int device_id,
    unsigned long long seed
);

#ifdef __cplusplus
}
#endif
