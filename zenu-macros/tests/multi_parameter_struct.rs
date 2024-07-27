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
    pub conv2d: Conv2d<T, D>,
    pub max_pool: MaxPool2d<T>,
}

#[derive(Parameters)]
struct LinearBlock<T: Num, D: Device> {
    pub linear: Linear<T, D>,
}

#[derive(Parameters)]
struct ConvNet<T: Num, D: Device> {
    pub conv_block: ConvBlock<T, D>,
    pub linear_block: LinearBlock<T, D>,
}

impl<T: Num, D: Device> ConvNet<T, D> {
    fn new() -> Self
    where
        StandardNormal: Distribution<T>,
    {
        Self {
            conv_block: ConvBlock {
                conv2d: Conv2d::new(3, 3, (1, 1), (1, 1), (1, 1), true),
                max_pool: MaxPool2d::new((2, 2), (2, 2), (0, 0)),
            },
            linear_block: LinearBlock {
                linear: Linear::new(2, 2, true),
            },
        }
    }
}

#[test]
fn multi_params() {
    let model = ConvNet::<f32, Cpu>::new();
    let conv_fileter = model.conv_block.conv2d.filter.clone();
    let conv_bias = model.conv_block.conv2d.bias.clone();
    let conv_bias = conv_bias.unwrap();
    let linear_weight = model.linear_block.linear.weight.clone();
    let linear_bias = model.linear_block.linear.bias.clone();
    let linear_bias = linear_bias.unwrap();

    let weights = model.weights();
    let biases = model.biases();
    let parameteers = model.parameters();
    assert_eq!(weights.len(), 2);
    assert_eq!(biases.len(), 2);
    assert_eq!(parameteers.len(), 4);

    assert_val_eq!(
        weights["conv_block.conv2d.conv2d.filter"].clone(),
        conv_fileter.get_data(),
        1e-6
    );
    assert_val_eq!(
        biases["conv_block.conv2d.conv2d.bias"].clone(),
        conv_bias.get_data(),
        1e-6
    );
    assert_val_eq!(
        weights["linear_block.linear.linear.weight"].clone(),
        linear_weight.get_data(),
        1e-6
    );
    assert_val_eq!(
        biases["linear_block.linear.linear.bias"].clone(),
        linear_bias.get_data(),
        1e-6
    );
}
