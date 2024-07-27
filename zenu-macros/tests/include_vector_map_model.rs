use std::collections::HashMap;

use zenu_layer::layers::{conv2d::Conv2d, linear::Linear, max_pool_2d::MaxPool2d};
use zenu_macros::Parameters;
use zenu_matrix::{device::Device, num::Num};

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
