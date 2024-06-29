use serde::{Deserialize, Serialize};

use crate::device::cpu::Cpu;
use crate::device::DeviceBase;
use crate::dim::{DimDyn, DimTrait};
use crate::matrix::{Matrix, Owned, Ptr};
use crate::num::Num;

// 中間構造体
#[derive(Serialize, Deserialize)]
struct MatrixSerializeData<T> {
    shape: Vec<usize>,
    stride: Vec<usize>,
    data: Vec<T>,
    data_type: String,
    ptr_offset: usize,
}

impl<T, D> Matrix<Owned<T>, DimDyn, D>
where
    T: Num,
    D: DeviceBase,
{
    fn to_serialize_data(&self) -> MatrixSerializeData<T> {
        let shape = self.shape().slice().to_vec();
        let stride = self.stride().slice().to_vec();
        let data = self
            .clone()
            .to::<Cpu>()
            .reshape([self.shape().num_elm()])
            .to_vec();
        let data_type = std::any::type_name::<T>().to_string();

        let ptr_offset = self.offset();

        MatrixSerializeData {
            shape,
            stride,
            data,
            data_type,
            ptr_offset,
        }
    }
}

impl<T> Matrix<Owned<T>, DimDyn, Cpu>
where
    T: Num,
{
    fn from_serialize_data(data: MatrixSerializeData<T>) -> Result<Self, String> {
        if std::any::type_name::<T>() != data.data_type {
            return Err("Data type mismatch".to_string());
        }

        let shape = DimDyn::from(&data.shape as &[usize]);
        let stride = DimDyn::from(&data.stride as &[usize]);

        let mut data_cloned = data.data.clone();
        let ptr =
            Ptr::<Owned<T>, Cpu>::new(data_cloned.as_mut_ptr(), data.data.len(), data.ptr_offset);
        std::mem::forget(data_cloned);

        Ok(Matrix::new(ptr, shape, stride))
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
        let data = MatrixSerializeData::<T>::deserialize(deserializer)?;
        let cpu_matrix =
            Matrix::<Owned<T>, DimDyn, Cpu>::from_serialize_data(data).map_err(|e| {
                serde::de::Error::custom(format!("Failed to deserialize matrix: {}", e))
            })?;
        Ok(cpu_matrix.to::<D>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;
    use zenu_test::assert_mat_eq_epsilon;

    #[test]
    fn test_matrix_serialization_deserialization() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape: Vec<usize> = vec![2, 3];
        let matrix: Matrix<Owned<f64>, DimDyn, Cpu> =
            Matrix::from_vec(data, DimDyn::from(&shape as &[usize]));

        let serialized = serde_json::to_string(&matrix).expect("Failed to serialize matrix");

        let deserialized: Matrix<Owned<f64>, DimDyn, Cpu> =
            serde_json::from_str(&serialized).expect("Failed to deserialize matrix");

        assert_eq!(matrix.shape(), deserialized.shape());
        assert_eq!(matrix.stride(), deserialized.stride());

        let original_data = matrix.to_vec();
        let deserialized_data = deserialized.to_vec();
        assert_eq!(original_data, deserialized_data);

        assert_mat_eq_epsilon!(matrix, deserialized, 1e-6);
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
    1.0,
    2.0,
    3.0,
    4.0
  ],
  "data_type": "f32",
  "ptr_offset": 0
}"#;

        assert_eq!(serialized, expected_json);
    }

    #[test]
    fn test_matrix_deserialization_error() {
        let invalid_json = r#"{
            "shape": [2, 2],
            "stride": [2, 1],
            "data": [1.0, 2.0, 3.0, 4.0],
            "data_type": "f64",
            "ptr_offset": 0
        }"#;

        let result: Result<Matrix<Owned<f32>, DimDyn, Cpu>, _> = serde_json::from_str(invalid_json);
        assert!(result.is_err());
    }
}
