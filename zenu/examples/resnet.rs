use zenu_autograd::{
    functions::{activation::relu::relu, flatten::flatten, pool2d::max_pool_2d},
    Variable,
};
use zenu_layer::{
    layers::{batch_norm_2d::BatchNorm2d, conv2d::Conv2d, linear::Linear},
    Module,
};
use zenu_matrix::device::Device;

struct ResBlock<D: Device> {
    conv1: Conv2d<f32, D>,
    batch_norm1: BatchNorm2d<f32, D>,
    conv2: Conv2d<f32, D>,
    batch_norm2: BatchNorm2d<f32, D>,
}

impl<D: Device> Module<f32, D> for ResBlock<D> {
    fn call(&self, inputs: Variable<f32, D>) -> Variable<f32, D> {
        let x = inputs.clone();
        let y = self.conv1.call(x.clone());
        let y = self.batch_norm1.call(y);
        let y = relu(y);
        let y = self.conv2.call(y);
        let y = self.batch_norm2.call(y);
        let y = y + x;
        relu(y)
    }
}

impl<D: Device> ResBlock<D> {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        padding: (usize, usize),
        stride: (usize, usize),
    ) -> Self {
        let conv1 = Conv2d::new(
            in_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            true,
        );
        let batch_norm1 = BatchNorm2d::new(out_channels, 0.9);
        let conv2 = Conv2d::new(
            out_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            true,
        );
        let batch_norm2 = BatchNorm2d::new(out_channels, 0.9);
        Self {
            conv1,
            batch_norm1,
            conv2,
            batch_norm2,
        }
    }
}

struct ResNet<D: Device> {
    conv1: Conv2d<f32, D>,
    res_block1: ResBlock<D>,
    res_block2: ResBlock<D>,
    linear: Linear<f32, D>,
}

impl<D: Device> Module<f32, D> for ResNet<D> {
    fn call(&self, inputs: Variable<f32, D>) -> Variable<f32, D> {
        let x = self.conv1.call(inputs.clone());
        // let x = max_pool_2d(x, (3, 3), (2, 2), (1, 1), config)
        let x = relu(x);
        let x = self.res_block1.call(x);
        let x = self.res_block2.call(x);
        let x = flatten(x);
        self.linear.call(x)
    }
}
