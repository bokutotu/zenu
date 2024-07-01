use std::cell::RefCell;

use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use zenu_autograd::{
    creator::{rand::normal, zeros::zeros},
    functions::conv2d::{conv2d, Conv2dConfigs},
    Variable,
};
use zenu_matrix::{device::Device, dim::DimTrait, nn::conv2d::conv2d_out_size, num::Num};

use crate::Layer;

#[derive(Serialize)]
pub struct Conv2d<T: Num, D: Device> {
    filter: Variable<T, D>,
    bias: Option<Variable<T, D>>,
    #[serde(skip)]
    config: RefCell<Option<Conv2dConfigs<T>>>,
    stride: (usize, usize),
    padding: (usize, usize),
}

impl<'de, T: Num, D: Device> Deserialize<'de> for Conv2d<T, D> {
    fn deserialize<Ds>(deserializer: Ds) -> Result<Self, Ds::Error>
    where
        Ds: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &["filter", "bias", "stride", "padding"];
        enum Field {
            Filter,
            Bias,
            Stride,
            Padding,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<Ds>(deserializer: Ds) -> Result<Field, Ds::Error>
            where
                Ds: serde::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> serde::de::Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter.write_str("`filter` or `bias` or `stride` or `padding`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "filter" => Ok(Field::Filter),
                            "bias" => Ok(Field::Bias),
                            "stride" => Ok(Field::Stride),
                            "padding" => Ok(Field::Padding),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct Conv2dVisitor<T: Num, D: Device> {
            marker: std::marker::PhantomData<Conv2d<T, D>>,
        }

        impl<'de, T: Num, D: Device> serde::de::Visitor<'de> for Conv2dVisitor<T, D> {
            type Value = Conv2d<T, D>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Conv2d")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Conv2d<T, D>, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let filter = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let bias = seq.next_element()?;
                let stride = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                let padding = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                Ok(Conv2d {
                    filter,
                    bias,
                    config: RefCell::new(None),
                    stride,
                    padding,
                })
            }

            fn visit_map<A>(self, mut map: A) -> Result<Conv2d<T, D>, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut filter = None;
                let mut bias = None;
                let mut stride = None;
                let mut padding = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Filter => {
                            if filter.is_some() {
                                return Err(serde::de::Error::duplicate_field("filter"));
                            }
                            filter = Some(map.next_value()?);
                        }
                        Field::Bias => {
                            if bias.is_some() {
                                return Err(serde::de::Error::duplicate_field("bias"));
                            }
                            bias = map.next_value()?;
                        }
                        Field::Stride => {
                            if stride.is_some() {
                                return Err(serde::de::Error::duplicate_field("stride"));
                            }
                            stride = Some(map.next_value()?);
                        }
                        Field::Padding => {
                            if padding.is_some() {
                                return Err(serde::de::Error::duplicate_field("padding"));
                            }
                            padding = Some(map.next_value()?);
                        }
                    }
                }
                let filter = filter.ok_or_else(|| serde::de::Error::missing_field("filter"))?;
                let stride = stride.ok_or_else(|| serde::de::Error::missing_field("stride"))?;
                let padding = padding.ok_or_else(|| serde::de::Error::missing_field("padding"))?;
                Ok(Conv2d {
                    filter,
                    bias,
                    config: RefCell::new(None),
                    stride,
                    padding,
                })
            }
        }

        deserializer.deserialize_struct(
            "Conv2d",
            FIELDS,
            Conv2dVisitor::<T, D> {
                marker: std::marker::PhantomData,
            },
        )
    }
}

impl<T: Num, D: Device> Layer<T, D> for Conv2d<T, D> {
    fn call(&self, input: Variable<T, D>) -> Variable<T, D> {
        if self.config.borrow().is_none() {
            let input_shape = input.get_data().shape();
            let filter_shape = self.filter.get_data().shape();
            let output_shape = conv2d_out_size(
                &input_shape.slice(),
                &filter_shape.slice(),
                self.padding,
                self.stride,
            );
            let config = Conv2dConfigs::new(
                input_shape.into(),
                output_shape.into(),
                filter_shape.into(),
                self.stride,
                self.padding,
                20,
            );
            *self.config.borrow_mut() = Some(config);
        }
        self.shape_check(&input);
        conv2d(
            input,
            self.filter.clone(),
            self.stride,
            self.padding,
            self.bias.clone(),
            Some(self.config.borrow().as_ref().unwrap().clone()),
        )
    }

    fn parameters(&self) -> Vec<Variable<T, D>> {
        let mut params = vec![self.filter.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn load_parameters(&mut self, parameters: &[Variable<T, D>]) {
        self.filter = parameters[0].clone();
        if parameters.len() > 1 {
            self.bias = Some(parameters[1].clone());
        }
    }

    fn shape_check(&self, input: &Variable<T, D>) {
        let input_shape = input.get_data().shape();
        let filter_shape = self.filter.get_data().shape();
        let bias_shape = self.bias.as_ref().map(|b| b.get_data().shape());

        assert_eq!(input_shape.len(), 4);
        assert_eq!(filter_shape.len(), 4);
        assert_eq!(input_shape[1], filter_shape[1]);
        assert_eq!(
            filter_shape[0],
            bias_shape.map_or(filter_shape[1], |b| b[1])
        );
    }
}

impl<T: Num, D: Device> Conv2d<T, D> {
    #[must_use]
    pub fn new(
        input_channel: usize,
        output_channel: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let filter_shape = [output_channel, input_channel, kernel_size.0, kernel_size.1];
        let bias = if bias {
            Some(zeros([1, output_channel, 1, 1]))
        } else {
            None
        };
        let filter = normal(T::zero(), T::one(), None, filter_shape);
        Conv2d {
            filter,
            bias,
            config: RefCell::new(None),
            stride,
            padding,
        }
    }
}

#[cfg(test)]
mod conv2d {
    use zenu_matrix::device::cpu::Cpu;

    use zenu_test::assert_mat_eq_epsilon;

    use crate::{layers::conv2d::Conv2d, Layer};

    #[test]
    fn conv2d() {
        let input = zenu_autograd::creator::rand::normal(0.0, 1.0, Some(42), &[2, 3, 5, 5]);
        let conv2d = Conv2d::new(3, 4, (3, 3), (1, 1), (1, 1), true);
        let output = conv2d.call(input);

        let conv2d_params = conv2d.parameters();

        let conv_2d_json = serde_json::to_string(&conv2d).unwrap();

        let deserialized_conv2d: Conv2d<f32, Cpu> = serde_json::from_str(&conv_2d_json).unwrap();

        let de_parames = deserialized_conv2d.parameters();

        assert_eq!(conv2d_params.len(), de_parames.len());
        for (p1, p2) in conv2d_params.iter().zip(de_parames.iter()) {
            assert_mat_eq_epsilon!(p1.get_data(), p2.get_data(), 1e-6);
        }
    }
}
