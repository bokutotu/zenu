use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    dim::DimTrait,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    num::Num,
    operation::{
        copy_from::CopyFrom,
        mul::Gemm,
        reshape::{Reshape, ReshapeMut},
        transpose::{Transpose, TransposeInplace},
    },
};

use crate::{creator::zeros::zeros, Function, Variable, VariableWeak};

use self::{
    conv2d_impl::{conv2d_inner, conv2d_out_size},
    deconv2_impl::{deconv2d_inner, deconv2d_out_size},
    im2col::im2col,
};

mod col2im;
mod conv2d_impl;
mod deconv2_impl;
mod im2col;

struct Conv2d<T: Num> {
    kernel: Variable<T>,
    input: Variable<T>,
    stride: (usize, usize),
    pad: (usize, usize),
    output: VariableWeak<T>,
}

struct Deconv2d<T: Num> {
    kernel: Variable<T>,
    input: Variable<T>,
    stride: (usize, usize),
    pad: (usize, usize),
    output: VariableWeak<T>,
}

struct Conv2dGrad<T: Num> {
    kernel: Variable<T>,
    input: Variable<T>,
    gradient_output: Variable<T>,
    stride: (usize, usize),
    pad: (usize, usize),
    output: VariableWeak<T>,
}

impl<T: Num> Function<T> for Conv2d<T> {
    fn forward(&self) {
        let output = conv2d_inner(
            self.input.get_data().to_view(),
            self.kernel.get_data().to_view(),
            self.pad,
            self.stride,
        );
        self.output
            .upgrade()
            .unwrap()
            .get_data_mut()
            .copy_from(&output);
    }

    fn backward(&self) {
        let g_input = deconv2d(
            self.output.upgrade().unwrap().get_grad().unwrap(),
            self.kernel.clone(),
            self.stride,
            self.pad,
        );
        let gw = conv2d_grad(
            self.input.clone(),
            self.output.upgrade().unwrap().get_grad().unwrap(),
            self.kernel.clone(),
            self.stride,
            self.pad,
        );
        self.input.set_grad(g_input);
        self.kernel.set_grad(gw);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.kernel.clone(), self.input.clone()]
    }
}

impl<T: Num> Function<T> for Deconv2d<T> {
    fn forward(&self) {
        let output = deconv2d_inner(
            self.input.get_data().to_view(),
            self.kernel.get_data().to_view(),
            self.pad,
            self.stride,
        );
        self.output
            .upgrade()
            .unwrap()
            .get_data_mut()
            .copy_from(&output);
    }

    fn backward(&self) {
        let g_input = conv2d(
            self.output.upgrade().unwrap().get_grad().unwrap(),
            self.kernel.clone(),
            self.stride,
            self.pad,
        );
        let gw = conv2d_grad(
            self.output.upgrade().unwrap().get_grad().unwrap(),
            self.input.clone(),
            self.kernel.clone(),
            self.stride,
            self.pad,
        );
        self.input.set_grad(g_input);
        self.kernel.set_grad(gw);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.kernel.clone(), self.input.clone()]
    }
}

impl<T: Num> Function<T> for Conv2dGrad<T> {
    fn forward(&self) {
        let input = self.input.get_data();
        let kernel_shape = self.kernel.get_data().shape();
        let col = im2col(
            input.to_view(),
            (kernel_shape[2], kernel_shape[3]),
            self.stride,
            self.pad,
        );
        let gradient_output = self.gradient_output.get_data();
        let grad_output_shape = gradient_output.shape();
        let grad_output_num_elm = grad_output_shape.num_elm();
        let gradient_output_transpose = gradient_output.transpose_swap_index_inplace(0, 1);
        let mut gradient_output_transose_reshape = gradient_output_transpose.reshape([
            grad_output_shape[1],
            grad_output_num_elm / grad_output_shape[1],
        ]);
        gradient_output_transose_reshape.transpose();

        let output = self.output.upgrade().unwrap();
        let output_shape = output.get_data().shape();
        let mut output = output.get_data_mut();
        let mut output = output.reshape_mut([col.col.shape()[0], grad_output_shape[1]]);
        output
            .to_view_mut()
            .gemm(col.col.to_view(), gradient_output_transose_reshape);
        output.reshape(output_shape.slice());
    }

    fn backward(&self) {
        let g_input = deconv2d(
            self.gradient_output.clone(),
            self.output.upgrade().unwrap(),
            self.stride,
            self.pad,
        );
        let grad_grad_output = conv2d(
            self.input.clone(),
            self.output.upgrade().unwrap(),
            self.stride,
            self.pad,
        );
        self.input.set_grad(g_input);
        self.gradient_output.set_grad(grad_grad_output);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.kernel.clone(), self.input.clone()]
    }
}

pub fn conv2d<T: Num>(
    x: Variable<T>,
    kernel: Variable<T>,
    stride: (usize, usize),
    pad: (usize, usize),
) -> Variable<T> {
    let x_shape = x.get_data().shape();
    let kernel_shape = kernel.get_data().shape();
    let out_shape = conv2d_out_size(x_shape.slice(), kernel_shape.slice(), pad, stride);
    let output = zeros(out_shape);
    let conv2d = Conv2d {
        kernel,
        input: x,
        stride,
        pad,
        output: output.clone().downgrade(),
    };

    conv2d.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(conv2d))));

    output
}

pub fn deconv2d<T: Num>(
    x: Variable<T>,
    kernel: Variable<T>,
    stride: (usize, usize),
    pad: (usize, usize),
) -> Variable<T> {
    let x_shape = x.get_data().shape();
    let kernel_shape = kernel.get_data().shape();
    let out_shape = deconv2d_out_size(x_shape.slice(), kernel_shape.slice(), pad, stride);
    let output = zeros(out_shape);
    let deconv2d = Deconv2d {
        kernel,
        input: x,
        stride,
        pad,
        output: output.clone().downgrade(),
    };
    deconv2d.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(deconv2d))));
    output
}

pub fn conv2d_grad<T: Num>(
    x: Variable<T>,
    gradient_output: Variable<T>,
    kernel: Variable<T>,
    stride: (usize, usize),
    pad: (usize, usize),
) -> Variable<T> {
    let output = zeros(kernel.get_data().shape());
    let conv2d_grad = Conv2dGrad {
        kernel,
        input: x,
        gradient_output,
        stride,
        pad,
        output: output.clone().downgrade(),
    };
    conv2d_grad.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(conv2d_grad))));
    output
}

#[cfg(test)]
mod conv2d {
    use crate::creator::from_vec::from_vec;

    #[test]
    fn conv2d_2x3x5x5_image_4x3x3x3_kernel_1x1_stride_1x1_padding() {
        let kernel = (1..(4 * 3 * 3 * 3 + 1))
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let kernel = from_vec(kernel, [4, 3, 3, 3]);
        let image = (1..(2 * 3 * 5 * 5 + 1))
            .map(|x| x as f32)
            .collect::<Vec<f32>>();
        let image = from_vec(image, [2, 3, 5, 5]);
        let output = super::conv2d(image, kernel, (1, 1), (1, 1));
        output.backward();
    }
}
