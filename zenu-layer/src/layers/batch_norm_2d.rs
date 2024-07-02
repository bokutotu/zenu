use serde::{ser::SerializeStruct, Deserialize, Serialize};
use zenu_autograd::{
    creator::{ones::ones, zeros::zeros},
    functions::batch_norm::{batch_norm_2d, BatchNorm2dAutoGradConfig},
    Variable,
};
use zenu_matrix::{
    device::Device,
    dim::{DimDyn, DimTrait},
    num::Num,
};

use crate::Layer;

pub struct BatchNorm2d<T: Num, D: Device> {
    config: BatchNorm2dAutoGradConfig<T>,
    momentum: f64,
    input_shape: DimDyn,
    scale: Variable<T, D>,
    bias: Variable<T, D>,
    mean: Variable<T, D>,
    variance: Variable<T, D>,
}

impl<T: Num, D: Device> Serialize for BatchNorm2d<T, D> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("BatchNorm2d", 6)?;
        state.serialize_field("momentum", &self.momentum)?;
        state.serialize_field("input_shape", &self.input_shape)?;
        state.serialize_field("scale", &self.scale)?;
        state.serialize_field("bias", &self.bias)?;
        state.serialize_field("mean", &self.mean)?;
        state.serialize_field("variance", &self.variance)?;

        state.end()
    }
}

impl<'de, T: Num + Deserialize<'de>, D: Device> Deserialize<'de> for BatchNorm2d<T, D> {
    fn deserialize<Ds: serde::Deserializer<'de>>(deserializer: Ds) -> Result<Self, Ds::Error> {
        const FIELDS: &[&str] = &[
            "momentum",
            "input_shape",
            "scale",
            "bias",
            "mean",
            "variance",
        ];

        enum Field {
            Momentum,
            InputShape,
            Scale,
            Bias,
            Mean,
            Variance,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> serde::de::Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter.write_str("field identifier")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "momentum" => Ok(Field::Momentum),
                            "input_shape" => Ok(Field::InputShape),
                            "scale" => Ok(Field::Scale),
                            "bias" => Ok(Field::Bias),
                            "mean" => Ok(Field::Mean),
                            "variance" => Ok(Field::Variance),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct BatchNorm2dVisitor<T: Num, D: Device> {
            marker: std::marker::PhantomData<BatchNorm2d<T, D>>,
        }

        impl<'de, T: Num + Deserialize<'de>, D: Device> serde::de::Visitor<'de>
            for BatchNorm2dVisitor<T, D>
        {
            type Value = BatchNorm2d<T, D>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct BatchNorm2d")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let momentum = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let input_shape: Vec<usize> = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let scale = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                let bias = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(3, &self))?;
                let mean = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(4, &self))?;
                let variance = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(5, &self))?;

                let config = BatchNorm2dAutoGradConfig::new(&input_shape);
                let input_shape = DimDyn::from(&input_shape as &[usize]);

                Ok(BatchNorm2d {
                    config,
                    momentum,
                    input_shape,
                    scale,
                    bias,
                    mean,
                    variance,
                })
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut momentum = None;
                let mut input_shape = None;
                let mut scale = None;
                let mut bias = None;
                let mut mean = None;
                let mut variance = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Momentum => {
                            if momentum.is_some() {
                                return Err(serde::de::Error::duplicate_field("momentum"));
                            }
                            momentum = Some(map.next_value()?);
                        }
                        Field::InputShape => {
                            if input_shape.is_some() {
                                return Err(serde::de::Error::duplicate_field("input_shape"));
                            }
                            input_shape = Some(map.next_value()?);
                        }
                        Field::Scale => {
                            if scale.is_some() {
                                return Err(serde::de::Error::duplicate_field("scale"));
                            }
                            scale = Some(map.next_value()?);
                        }
                        Field::Bias => {
                            if bias.is_some() {
                                return Err(serde::de::Error::duplicate_field("bias"));
                            }
                            bias = Some(map.next_value()?);
                        }
                        Field::Mean => {
                            if mean.is_some() {
                                return Err(serde::de::Error::duplicate_field("mean"));
                            }
                            mean = Some(map.next_value()?);
                        }
                        Field::Variance => {
                            if variance.is_some() {
                                return Err(serde::de::Error::duplicate_field("variance"));
                            }
                            variance = Some(map.next_value()?);
                        }
                    }
                }

                let momentum =
                    momentum.ok_or_else(|| serde::de::Error::missing_field("momentum"))?;
                let input_shape: Vec<usize> =
                    input_shape.ok_or_else(|| serde::de::Error::missing_field("input_shape"))?;
                let scale = scale.ok_or_else(|| serde::de::Error::missing_field("scale"))?;
                let bias = bias.ok_or_else(|| serde::de::Error::missing_field("bias"))?;
                let mean = mean.ok_or_else(|| serde::de::Error::missing_field("mean"))?;
                let variance =
                    variance.ok_or_else(|| serde::de::Error::missing_field("variance"))?;

                let config = BatchNorm2dAutoGradConfig::new(&input_shape);
                let input_shape = DimDyn::from(&input_shape as &[usize]);

                Ok(BatchNorm2d {
                    config,
                    momentum,
                    input_shape,
                    scale,
                    bias,
                    mean,
                    variance,
                })
            }
        }

        deserializer.deserialize_struct(
            "BatchNorm2d",
            FIELDS,
            BatchNorm2dVisitor {
                marker: std::marker::PhantomData,
            },
        )
    }
}

impl<T: Num, D: Device> Layer<T, D> for BatchNorm2d<T, D> {
    fn call(&self, input: Variable<T, D>) -> Variable<T, D> {
        batch_norm_2d(
            input,
            self.scale.clone(),
            self.bias.clone(),
            self.mean.clone(),
            self.variance.clone(),
            self.momentum,
            self.config.clone(),
        )
    }

    fn parameters(&self) -> Vec<Variable<T, D>> {
        vec![
            self.scale.clone(),
            self.bias.clone(),
            self.mean.clone(),
            self.variance.clone(),
        ]
    }

    fn load_parameters(&mut self, parameters: &[Variable<T, D>]) {
        self.scale = parameters[0].clone();
        self.bias = parameters[1].clone();
        self.mean = parameters[2].clone();
        self.variance = parameters[3].clone();
    }

    fn shape_check(&self, input: &Variable<T, D>) {
        let input_shape = input.get_data().shape();
        let scale_shape = self.scale.get_data().shape();
        let bias_shape = self.bias.get_data().shape();
        let mean_shape = self.mean.get_data().shape();
        let variance_shape = self.variance.get_data().shape();

        assert_eq!(input_shape.len(), 4);
        assert_eq!(scale_shape.len(), 1);
        assert_eq!(bias_shape.len(), 1);
        assert_eq!(mean_shape.len(), 1);
        assert_eq!(variance_shape.len(), 1);
        assert_eq!(scale_shape[0], input_shape[1]);
        assert_eq!(bias_shape[0], input_shape[1]);
        assert_eq!(mean_shape[0], input_shape[1]);
        assert_eq!(variance_shape[0], input_shape[1]);
    }
}

impl<T: Num, D: Device> BatchNorm2d<T, D> {
    #[must_use]
    pub fn new(input_shape: DimDyn, momentum: f64) -> Self {
        let scale = ones([input_shape[1]]);
        let bias = zeros([input_shape[1]]);
        let mean = zeros([input_shape[1]]);
        let variance = ones([input_shape[1]]);
        let config = BatchNorm2dAutoGradConfig::new(input_shape.slice());
        Self {
            config,
            momentum,
            input_shape,
            scale,
            bias,
            mean,
            variance,
        }
    }
}
