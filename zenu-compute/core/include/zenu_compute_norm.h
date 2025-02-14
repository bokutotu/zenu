/**
 * @file zenu_compute_norm.h
 * @brief バッチ正規化（Batch Normalization）に関する API を提供します.
 *
 * このヘッダファイルでは、バッチ正規化のハンドル作成、初期化、
 * 推論および学習時のフォワード演算、逆伝播演算、ならびに必要な
 * ワークスペースサイズの取得関数が宣言されています.
 */

#pragma once

#include "zenu_compute_type.h"
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief バッチ正規化ハンドルの不透明な構造体.
 */
typedef struct ZenuComputeBatchNorm ZenuComputeBatchNorm;

/**
 * @brief バッチ正規化ハンドルの作成.
 *
 * 新しいバッチ正規化ハンドルを生成します。
 *
 * @param[out] batchnorm 生成されたバッチ正規化ハンドルのアドレスを返します.
 */
void zenu_compute_create_batchnorm(ZenuComputeBatchNorm** batchnorm);

/**
 * @brief バッチ正規化ハンドルの破棄.
 *
 * 指定されたバッチ正規化ハンドルおよび関連リソースを解放します.
 *
 * @param[in] batchnorm 破棄対象のバッチ正規化ハンドル.
 */
void zenu_compute_destroy_batchnorm(ZenuComputeBatchNorm* batchnorm);

/**
 * @brief バッチ正規化ハンドルの初期化.
 *
 * 入力データの形状やデータ型、学習モードの有無などのパラメータに基づいて、
 * バッチ正規化ハンドルを初期化します.
 *
 * @param[in]  batchnorm バッチ正規化ハンドル.
 * @param[in]  shape     各次元サイズを格納した配列.
 * @param[in]  dim       shape 配列の要素数.
 * @param[in]  type      データ型 (例: ZenuDataType::f32)。
 * @param[in]  is_train  学習モードの場合は true、推論モードの場合は false.
 * @return ZenuStatus 成功時は Success、それ以外はエラーコードを返します.
 */
ZenuStatus zenu_compute_init_batchnorm(
    ZenuComputeBatchNorm* batchnorm,
    size_t* shape,
    size_t dim,
    ZenuDataType type,
    bool is_train
);

/**
 * @brief 推論時のバッチ正規化フォワード演算を実行.
 *
 * 推論モードにおいて、入力データに対してバッチ正規化を実行します。
 *
 * @param[in]  batchnorm     バッチ正規化ハンドル.
 * @param[in]  input         入力データへのポインタ.
 * @param[in]  scale         スケールパラメータへのポインタ.
 * @param[in]  bias          バイアスパラメータへのポインタ.
 * @param[in]  mean          平均値パラメータへのポインタ.
 * @param[in]  inv_variance  分散の逆数パラメータへのポインタ.
 * @param[out] output        出力データへのポインタ.
 * @param[out] workspace     ワークスペース領域へのポインタ.
 * @return ZenuStatus 成功時は Success、それ以外はエラーコードを返します.
 */
ZenuStatus zenu_compute_forward_batchnorm_inference(
    ZenuComputeBatchNorm* batchnorm,
    const void* input,
    const void* scale,
    const void* bias,
    const void* mean,
    const void* inv_variance,
    void* output,
    void* workspace
);

/**
 * @brief 学習時のバッチ正規化フォワード演算を実行.
 *
 * 学習モードにおいて、入力データに対してバッチ正規化を実行し、
 * 計算された平均値および分散の逆数を後方伝播用として出力します。
 *
 * @param[in]  batchnorm     バッチ正規化ハンドル.
 * @param[in]  input         入力データへのポインタ.
 * @param[in]  scale         スケールパラメータへのポインタ.
 * @param[in]  bias          バイアスパラメータへのポインタ.
 * @param[out] output        出力データへのポインタ.
 * @param[out] mean          計算された平均値へのポインタ（後方伝播用）.
 * @param[out] inv_variance  計算された分散の逆数へのポインタ（後方伝播用）.
 * @param[out] workspace     ワークスペース領域へのポインタ.
 * @return ZenuStatus 成功時は Success、それ以外はエラーコードを返します.
 */
ZenuStatus zenu_compute_forward_batchnorm_train(
    ZenuComputeBatchNorm* batchnorm,
    const void* input,
    const void* scale,
    const void* bias,
    void* output,
    void* mean,
    void* inv_variance,
    void* workspace
);

/**
 * @brief バッチ正規化の逆伝播演算を実行.
 *
 * 出力勾配と入力データ、パラメータ情報から、入力に対する勾配、
 * スケールおよびバイアスに対する勾配を計算します.
 *
 * @param[in]  batchnorm     バッチ正規化ハンドル.
 * @param[in]  d_output      出力の勾配データへのポインタ.
 * @param[in]  input         入力データへのポインタ.
 * @param[in]  scale         スケールパラメータへのポインタ.
 * @param[in]  mean          平均値パラメータへのポインタ.
 * @param[in]  inv_variance  分散の逆数パラメータへのポインタ.
 * @param[out] d_input       入力データに対する勾配へのポインタ.
 * @param[out] d_scale       スケールパラメータに対する勾配へのポインタ.
 * @param[out] d_bias        バイアスパラメータに対する勾配へのポインタ.
 * @param[out] workspace     ワークスペース領域へのポインタ.
 * @return ZenuStatus 成功時は Success、それ以外はエラーコードを返します.
 */
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
);

/**
 * @brief フォワード演算に必要なワークスペースサイズを取得.
 *
 * バッチ正規化のフォワード演算を実行するために必要なワークスペースの
 * バイト数を返します.
 *
 * @param[in] batchnorm バッチ正規化ハンドル.
 * @return size_t 必要なワークスペースのバイト数.
 */
size_t zenu_compute_batchnorm_forward_get_workspace_bytes(
    ZenuComputeBatchNorm* batchnorm
);

/**
 * @brief 逆伝播演算に必要なワークスペースサイズを取得.
 *
 * バッチ正規化の逆伝播演算を実行するために必要なワークスペースの
 * バイト数を返します.
 *
 * @param[in] batchnorm バッチ正規化ハンドル.
 * @return size_t 必要なワークスペースのバイト数.
 */
size_t zenu_compute_batchnorm_backward_get_workspace_bytes(
    ZenuComputeBatchNorm* batchnorm
);

#ifdef __cplusplus
} // extern "C"
#endif

