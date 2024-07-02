use serde::{
    de::{Deserialize, Visitor},
    ser::{Serialize, SerializeStruct},
};

use crate::device::cpu::Cpu;
use crate::device::Device;
use crate::dim::{DimDyn, DimTrait};
use crate::matrix::{Matrix, Owned, Ptr, Repr};
use crate::num::Num;

impl<R: Repr, D: Device> Serialize for Matrix<R, DimDyn, D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let shape = self.shape().slice().to_vec();
        let stride = self.stride().slice().to_vec();
        let data = self
            .new_matrix()
            .clone()
            .to::<Cpu>()
            .reshape([self.shape().num_elm()])
            .to_vec();
        let data_type = std::any::type_name::<R::Item>().to_string();
        let ptr_offset = self.offset();

        let mut state = serializer.serialize_struct("Matrix", 5)?;

        state.serialize_field("shape", &shape)?;
        state.serialize_field("stride", &stride)?;
        state.serialize_field("data", &data)?;
        state.serialize_field("data_type", &data_type)?;
        state.serialize_field("ptr_offset", &ptr_offset)?;

        state.end()
    }
}

impl<'de, T: Num + Deserialize<'de>, D: Device> Deserialize<'de> for Matrix<Owned<T>, DimDyn, D> {
    fn deserialize<Ds>(deserializer: Ds) -> Result<Self, Ds::Error>
    where
        Ds: serde::Deserializer<'de>,
    {
        enum Field {
            Shape,
            Stride,
            Data,
            DataType,
            PtrOffset,
        }

        const FIELDS: &[&str] = &["shape", "stride", "data", "data_type", "ptr_offset"];

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> serde::de::Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter
                            .write_str("`shape`, `stride`, `data`, `data_type` or `ptr_offset`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "shape" => Ok(Field::Shape),
                            "stride" => Ok(Field::Stride),
                            "data" => Ok(Field::Data),
                            "data_type" => Ok(Field::DataType),
                            "ptr_offset" => Ok(Field::PtrOffset),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct MatrixVisitor<T: Num, D: Device>(std::marker::PhantomData<(T, D)>);

        impl<'de, T: Num + Deserialize<'de>, D: Device> Visitor<'de> for MatrixVisitor<T, D> {
            type Value = Matrix<Owned<T>, DimDyn, D>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Matrix")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut shape = None;
                let mut stride = None;
                let mut data = None;
                let mut data_type = None;
                let mut ptr_offset = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Shape => {
                            if shape.is_some() {
                                return Err(serde::de::Error::duplicate_field("shape"));
                            }
                            shape = Some(map.next_value()?);
                        }
                        Field::Stride => {
                            if stride.is_some() {
                                return Err(serde::de::Error::duplicate_field("stride"));
                            }
                            stride = Some(map.next_value()?);
                        }
                        Field::Data => {
                            if data.is_some() {
                                return Err(serde::de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                        Field::DataType => {
                            if data_type.is_some() {
                                return Err(serde::de::Error::duplicate_field("data_type"));
                            }
                            data_type = Some(map.next_value()?);
                        }
                        Field::PtrOffset => {
                            if ptr_offset.is_some() {
                                return Err(serde::de::Error::duplicate_field("ptr_offset"));
                            }
                            ptr_offset = Some(map.next_value()?);
                        }
                    }
                }

                let shape: Vec<usize> =
                    shape.ok_or_else(|| serde::de::Error::missing_field("shape"))?;
                let stride: Vec<usize> =
                    stride.ok_or_else(|| serde::de::Error::missing_field("stride"))?;
                let data: Vec<T> = data.ok_or_else(|| serde::de::Error::missing_field("data"))?;
                let data_type: String =
                    data_type.ok_or_else(|| serde::de::Error::missing_field("data_type"))?;
                let ptr_offset =
                    ptr_offset.ok_or_else(|| serde::de::Error::missing_field("ptr_offset"))?;

                if std::any::type_name::<T>() != data_type {
                    return Err(serde::de::Error::custom("Data type mismatch"));
                }

                let shape = DimDyn::from(&shape as &[usize]);
                let stride = DimDyn::from(&stride as &[usize]);

                let mut data_cloned = data.clone();
                let ptr =
                    Ptr::<Owned<T>, Cpu>::new(data_cloned.as_mut_ptr(), data.len(), ptr_offset);
                std::mem::forget(data_cloned);

                let mat = Matrix::new(ptr, shape, stride);
                let mat_d: Matrix<Owned<T>, DimDyn, D> = mat.to();
                Ok(mat_d)
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let shape: Vec<usize> = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &""))?;
                let stride: Vec<usize> = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &""))?;
                let data: Vec<T> = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &""))?;
                let data_type: String = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(3, &""))?;
                let ptr_offset = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(4, &""))?;

                if std::any::type_name::<T>() != data_type {
                    return Err(serde::de::Error::custom("Data type mismatch"));
                }

                let shape = DimDyn::from(&shape as &[usize]);
                let stride = DimDyn::from(&stride as &[usize]);

                let mut data_cloned = data.clone();
                let ptr =
                    Ptr::<Owned<T>, Cpu>::new(data_cloned.as_mut_ptr(), data.len(), ptr_offset);
                std::mem::forget(data_cloned);

                let mat = Matrix::new(ptr, shape, stride);
                let mat_d: Matrix<Owned<T>, DimDyn, D> = mat.to();
                Ok(mat_d)
            }
        }

        deserializer.deserialize_struct(
            "Matrix",
            FIELDS,
            MatrixVisitor::<T, D>(std::marker::PhantomData),
        )
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
