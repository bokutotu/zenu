#pragma once

#include <vector>
#include "cudnn_frontend.h"
#include "zenu_compute_type.h"

namespace fe = cudnn_frontend;

/**
 * @brief from_shape 関数は、引数で与えられた配列 dims の先頭 n 個の要素から std::vector<int64_t> を生成します。
 * @param dims 長さ 8 の int64_t 配列
 * @return n 個の要素を格納した std::vector<int64_t>
 */
std::vector<size_t> from_shape(std::vector<size_t> dims);

/**
 * @brief from_shape 関数は、引数で与えられた配列 dims の先頭 n 個の要素から std::vector<int64_t> を生成します。
 * @param n 取得する要素数
 * @param dims 長さ 8 の int64_t 配列
 * @return n 個の要素を格納した std::vector<int64_t>
 */
std::vector<int64_t> default_stride(std::vector<size_t> shape);

/**
 * @brief get_data_type 関数は、ZenuDataType を cudnn_frontend::DataType_t に変換します。
 * @param data_type ZenuDataType 型の列挙値
 * @return 対応する cudnn_frontend::DataType_t
 * @throws std::runtime_error 不明または未サポートの data_type が指定された場合にスローされます
 */
fe::DataType_t get_data_type(ZenuDataType data_type);

/**
 * @brief get_tensor_attributes 関数は、指定された形状とストライド、データ型を設定した cudnn_frontend::graph::Tensor_attributes を作成します。
 * @param shape テンソルの次元数とサイズを表す std::vector<size_t>
 * @param data_type テンソルのデータ型 (ZenuDataType)
 * @return 生成された cudnn_frontend::graph::Tensor_attributes オブジェクト
 */
fe::graph::Tensor_attributes get_tensor_attributes(std::vector<size_t> shape,
                                                   ZenuDataType data_type);

/**
 * @brief get_tensor_attributes_without_type 関数は、データ型を設定せずに cudnn_frontend::graph::Tensor_attributes を作成します。
 * @param shape テンソルの次元数とサイズを表す std::vector<size_t>
 * @return データ型が設定されていない cudnn_frontend::graph::Tensor_attributes オブジェクト
 */
fe::graph::Tensor_attributes get_tensor_attributes_without_type(std::vector<size_t> shape);

/**
 * @brief build_and_check_graph はcudnnのグラフを構築し、チェックします。
 * @param handle cudnnハンドル
 * @return ZenuStatus 成功した場合はSuccess、エラーが発生した場合はCudnnError
 */
ZenuStatus build_and_check_graph(fe::graph::Graph& graph);

/**
 * @brief get_workspace_size はワークスペースのサイズを取得します。
 * @param workspace_size ワークスペースのサイズ
 * @return ZenuStatus 成功した場合はSuccess、エラーが発生した場合はCudnnError
 */
size_t get_workspace_size(fe::graph::Graph& graph);

