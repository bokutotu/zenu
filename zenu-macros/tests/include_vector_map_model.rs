use std::collections::HashMap;

use rand_distr::{Distribution, StandardNormal};
use zenu_layer::{
    layers::{conv2d::Conv2d, linear::Linear, max_pool_2d::MaxPool2d},
    Parameters,
};
use zenu_macros::Parameters;
use zenu_matrix::{
    device::{cpu::Cpu, Device},
    num::Num,
};
use zenu_test::assert_val_eq;

#[derive(Parameters)]
struct ConvBlock<T: Num, D: Device> {
    conv2d: Conv2d<T, D>,
    max_pool: MaxPool2d<T>,
}

#[derive(Parameters)]
struct LinearBlock<T: Num, D: Device> {
    linear: Linear<T, D>,
}

#[derive(Parameters)]
struct ConvNet<T: Num, D: Device> {
    conv_blocks: Vec<ConvBlock<T, D>>,
    linear_block: HashMap<String, LinearBlock<T, D>>,
}

impl<T: Num, D: Device> ConvNet<T, D> {
    fn new() -> Self
    where
        StandardNormal: Distribution<T>,
    {
        Self {
            conv_blocks: vec![
                ConvBlock {
                    conv2d: Conv2d::new(3, 3, (1, 1), (1, 1), (1, 1), true),
                    max_pool: MaxPool2d::new((2, 2), (2, 2), (0, 0)),
                },
                ConvBlock {
                    conv2d: Conv2d::new(3, 3, (1, 1), (1, 1), (1, 1), true),
                    max_pool: MaxPool2d::new((2, 2), (2, 2), (0, 0)),
                },
            ],
            linear_block: vec![
                (
                    String::from("linear1"),
                    LinearBlock {
                        linear: Linear::new(2, 2, true),
                    },
                ),
                (
                    String::from("linear2"),
                    LinearBlock {
                        linear: Linear::new(2, 2, true),
                    },
                ),
            ]
            .into_iter()
            .collect(),
        }
    }
}

#[test]
fn vec_map() {
    let model = ConvNet::<f32, Cpu>::new();
    let conv_fileter_0 = model.conv_blocks[0].conv2d.filter.clone();
    let conv_bias_0 = model.conv_blocks[0].conv2d.bias.clone();
    let conv_bias_0 = conv_bias_0.unwrap();
    let conv_fileter_1 = model.conv_blocks[1].conv2d.filter.clone();
    let conv_bias_1 = model.conv_blocks[1].conv2d.bias.clone();
    let conv_bias_1 = conv_bias_1.unwrap();
    let linear_weight_0 = model
        .linear_block
        .get("linear1")
        .unwrap()
        .linear
        .weight
        .clone();
    let linear_bias_0 = model
        .linear_block
        .get("linear1")
        .unwrap()
        .linear
        .bias
        .clone();
    let linear_bias_0 = linear_bias_0.unwrap();
    let linear_weight_1 = model
        .linear_block
        .get("linear2")
        .unwrap()
        .linear
        .weight
        .clone();
    let linear_bias_1 = model
        .linear_block
        .get("linear2")
        .unwrap()
        .linear
        .bias
        .clone();
    let linear_bias_1 = linear_bias_1.unwrap();

    let weights = model.weights();
    let biases = model.biases();
    let parameters = model.parameters();
    assert_eq!(weights.len(), 4);
    assert_eq!(biases.len(), 4);
    assert_eq!(parameters.len(), 8);

    assert_val_eq!(
        weights["conv_blocks.0.conv2d.conv2d.filter"].clone(),
        conv_fileter_0.get_data(),
        1e-6
    );
    assert_val_eq!(
        biases["conv_blocks.0.conv2d.conv2d.bias"].clone(),
        conv_bias_0.get_data(),
        1e-6
    );
    assert_val_eq!(
        weights["conv_blocks.1.conv2d.conv2d.filter"].clone(),
        conv_fileter_1.get_data(),
        1e-6
    );
    assert_val_eq!(
        biases["conv_blocks.1.conv2d.conv2d.bias"].clone(),
        conv_bias_1.get_data(),
        1e-6
    );
    assert_val_eq!(
        weights["linear_block.linear1.linear.linear.weight"].clone(),
        linear_weight_0.get_data(),
        1e-6
    );
    assert_val_eq!(
        biases["linear_block.linear1.linear.linear.bias"].clone(),
        linear_bias_0.get_data(),
        1e-6
    );
    assert_val_eq!(
        weights["linear_block.linear2.linear.linear.weight"].clone(),
        linear_weight_1.get_data(),
        1e-6
    );
    assert_val_eq!(
        biases["linear_block.linear2.linear.linear.bias"].clone(),
        linear_bias_1.get_data(),
        1e-6
    );
}
