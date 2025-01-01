#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CPU上で正規分布乱数を生成し、dst に書き込む
 *
 * @param dst          出力バッファ (float* or double*)
 * @param num_elements 要素数
 * @param mean         平均
 * @param stddev       標準偏差
 * @param data_type    f32 or f64
 * @return ZenuStatus  成功 or 各種エラー
 */
ZenuStatus zenu_compute_normal_distribution_cpu(void* dst,
                                                int   num_elements,
                                                float mean,
                                                float stddev,
                                                ZenuDataType data_type);

/**
 * @brief CPU上で一様分布乱数を生成し、dst に書き込む
 *
 * @param dst          出力バッファ (float* or double*)
 * @param num_elements 要素数
 * @param low          一様分布の下限
 * @param high         一様分布の上限
 * @param data_type    f32 or f64
 * @return ZenuStatus  成功 or 各種エラー
 */
ZenuStatus zenu_compute_uniform_distribution_cpu(void* dst,
                                                 int   num_elements,
                                                 float low,
                                                 float high,
                                                 ZenuDataType data_type);

ZenuStatus zenu_compute_normal_distribution_nvidia(void* dst,
                                            int num_elements,
                                            float mean,
                                            float stddev,
                                            ZenuDataType data_type,
                                            int device_id);

ZenuStatus zenu_compute_uniform_distribution_nvidia(void* dst,
                                             int num_elements,
                                             float low,
                                             float high,
                                             ZenuDataType data_type,
                                             int device_id);

#ifdef __cplusplus
}
#endif
