use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::str::FromStr;

use crate::device::cpu::Cpu;
use crate::device::DeviceBase;
use crate::dim::{DimDyn, DimTrait};
use crate::matrix::{Matrix, Owned};
use crate::num::Num;
use crate::shape_stride::ShapeStride;

// 中間構造体
#[derive(Serialize, Deserialize)]
struct MatrixSerializeData<T> {
    shape: Vec<usize>,
    stride: Vec<usize>,
    data: Vec<T>,
    data_type: String,
}

impl<T, D> Matrix<Owned<T>, DimDyn, D>
where
    T: Num,
    D: DeviceBase,
{
    // MatrixSerializeDataへの変換メソッド
    fn to_serialize_data(&self) -> MatrixSerializeData<T> {
        let shape = self.shape().slice().to_vec();
        let stride = self.stride().slice().to_vec();
        let data = self
            .clone()
            .to::<Cpu>()
            .reshape([self.shape().num_elm()])
            .to_vec();
        let data_type = std::any::type_name::<T>().to_string();

        MatrixSerializeData {
            shape,
            stride,
            data,
            data_type,
        }
    }

    // MatrixSerializeDataからMatrixを作成するメソッド
    fn from_serialize_data(data: MatrixSerializeData<T>) -> Result<Self, String> {
        if std::any::type_name::<T>() != data.data_type {
            return Err("Data type mismatch".to_string());
        }

        let shape = DimDyn::from(&data.shape as &[usize]);
        let stride = DimDyn::from(&data.stride as &[usize]);
        // let matrix = Self::from_vec(data.data, shape);
        //
        // Ok(matrix.with_stride(stride))
        let mut matrix = Matrix::from_vec(data.data, shape);
        let shape_stride = ShapeStride::new(shape, stride);
        matrix.update_shape_stride(shape_stride);
        Ok(matrix)
    }
}

impl<T, D> Serialize for Matrix<Owned<T>, DimDyn, D>
where
    T: Num,
    D: DeviceBase,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        self.to_serialize_data().serialize(serializer)
    }
}

impl<'de, T, D> Deserialize<'de> for Matrix<Owned<T>, DimDyn, D>
where
    T: Num,
    D: DeviceBase,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        let data = MatrixSerializeData::deserialize(deserializer)?;
        Self::from_serialize_data(data).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_matrix_serialization_deserialization() {
        // 1. テスト用のMatrixを作成
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape: Vec<usize> = vec![2, 3];
        let matrix: Matrix<Owned<f64>, DimDyn, Cpu> =
            Matrix::from_vec(data, DimDyn::from(&shape as &[usize]));

        // 2. Matrixをシリアライズ
        let serialized = serde_json::to_string(&matrix).expect("Failed to serialize matrix");

        // 3. シリアライズされた文字列をデシリアライズ
        let deserialized: Matrix<Owned<f64>, DimDyn, Cpu> =
            serde_json::from_str(&serialized).expect("Failed to deserialize matrix");

        // 4. 元のMatrixとデシリアライズされたMatrixを比較
        assert_eq!(matrix.shape(), deserialized.shape());
        assert_eq!(matrix.stride(), deserialized.stride());

        let original_data = matrix.to_vec();
        let deserialized_data = deserialized.to_vec();
        assert_eq!(original_data, deserialized_data);

        // 5. 値を個別に確認
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(matrix.index_item([i, j]), deserialized.index_item([i, j]));
            }
        }
    }

    #[test]
    fn test_matrix_serialization_format() {
        let data = vec![1., 2., 3., 4.];
        let shape: Vec<usize> = vec![2, 2];
        let matrix: Matrix<Owned<f32>, DimDyn, Cpu> =
            Matrix::from_vec(data, DimDyn::from(&shape as &[usize]));

        let serialized = serde_json::to_string_pretty(&matrix).expect("Failed to serialize matrix");

        // 期待されるJSON形式を確認
        let expected_json = r#"{
  "shape": [
    2,
    2
  ],
  "stride": [
    2,
    1
  ],
  "data": [
    1,
    2,
    3,
    4
  ],
  "data_type": "f32"
}"#;

        assert_eq!(serialized, expected_json);
    }

    #[test]
    fn test_matrix_deserialization_error() {
        // データ型が一致しない場合のテスト
        let invalid_json = r#"{
            "shape": [2, 2],
            "stride": [2, 1],
            "data": [1.0, 2.0, 3.0, 4.0],
            "data_type": "f64"
        }"#;

        let result: Result<Matrix<Owned<f64>, DimDyn, Cpu>, _> = serde_json::from_str(invalid_json);
        assert!(result.is_err());
    }
}
