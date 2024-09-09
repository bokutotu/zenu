use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    dim::{DimDyn, DimTrait},
    nn::pool2d::{
        max_pool_2d as mat_forward, max_pool_2d_grad as mat_backward, max_pool_2d_output_shape,
        Pool2dConfig as C,
    },
    num::Num,
};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

#[derive(Clone, Default)]
pub struct MaxPool2dConfig<T: Num> {
    pub config: Rc<RefCell<Option<C<T>>>>,
}

impl<T: Num> MaxPool2dConfig<T> {
    pub fn update(
        &self,
        input_shape: DimDyn,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        pad: (usize, usize),
    ) {
        let output_shape = max_pool_2d_output_shape(input_shape.slice(), kernel_size, stride, pad);
        let input_shape = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let output_shape = (
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
        );
        let config = C::new(kernel_size, stride, pad, input_shape, output_shape);
        *self.config.borrow_mut() = Some(config);
    }

    #[must_use]
    pub fn is_none(&self) -> bool {
        self.config.borrow().is_none()
    }
}

struct MaxPool2d<T: Num, D: Device> {
    config: MaxPool2dConfig<T>,
    input: Variable<T, D>,
    output: VariableWeak<T, D>,
    stride: (usize, usize),
    pad: (usize, usize),
    kernel_size: (usize, usize),
}

struct MaxPool2dBkwd<T: Num, D: Device> {
    config: MaxPool2dConfig<T>,
    input: Variable<T, D>,
    input_grad: VariableWeak<T, D>,
    output: Variable<T, D>,
    output_grad: Variable<T, D>,
    stride: (usize, usize),
    pad: (usize, usize),
    kernel_size: (usize, usize),
}

impl<T, D> Function<T, D> for MaxPool2d<T, D>
where
    T: Num,
    D: Device,
{
    fn forward(&self) {
        let config = self.config.config.clone();
        let config = config.borrow();
        let config = config.as_ref().unwrap();
        self.output
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&mat_forward(
                self.input.get_data().to_ref(),
                self.kernel_size,
                self.stride,
                self.pad,
                config,
            ));
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().unwrap();
        let input_grad = max_pool_2d_grad(
            self.input.clone(),
            output,
            output_grad,
            self.kernel_size,
            self.stride,
            self.pad,
            self.config.clone(),
        );
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

impl<T, D> Function<T, D> for MaxPool2dBkwd<T, D>
where
    T: Num,
    D: Device,
{
    fn forward(&self) {
        let config = self.config.config.clone();
        let config = config.borrow();
        let config = config.as_ref().unwrap();
        self.input_grad
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&mat_backward(
                self.input.get_data().to_ref(),
                self.output.get_data().to_ref(),
                self.output_grad.get_data().to_ref(),
                self.kernel_size,
                self.stride,
                self.pad,
                config,
            ));
    }

    fn backward(&self) {
        unimplemented!()
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.output.clone()]
    }
}

#[must_use]
pub fn max_pool_2d<T: Num, D: Device>(
    input: Variable<T, D>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    config: MaxPool2dConfig<T>,
) -> Variable<T, D> {
    let output_shape =
        max_pool_2d_output_shape(input.get_shape().slice(), kernel_size, stride, pad);

    let output = alloc(output_shape);

    if config.is_none() {
        config.update(input.get_shape(), kernel_size, stride, pad);
    }

    let function = MaxPool2d {
        config,
        input,
        output: output.clone().downgrade(),
        stride,
        pad,
        kernel_size,
    };
    function.forward();

    output.set_creator(Rc::new(RefCell::new(Box::new(function))));
    output
}
#[must_use]
pub fn max_pool_2d_grad<T: Num, D: Device>(
    input: Variable<T, D>,
    output: Variable<T, D>,
    output_grad: Variable<T, D>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    config: MaxPool2dConfig<T>,
) -> Variable<T, D> {
    let input_grad = alloc(input.get_shape().slice());

    let function = MaxPool2dBkwd {
        config,
        input,
        input_grad: input_grad.clone().downgrade(),
        output,
        output_grad,
        stride,
        pad,
        kernel_size,
    };

    function.forward();

    input_grad.set_creator(Rc::new(RefCell::new(Box::new(function))));
    input_grad
}

#[cfg(test)]
mod pool2d {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::{creator::from_vec::from_vec, functions::pool2d::MaxPool2dConfig};

    use super::max_pool_2d;

    #[expect(clippy::unreadable_literal, clippy::too_many_lines)]
    fn _pool2d<D: Device>() {
        let input = vec![
            -1.1258398,
            -1.1523602,
            -0.25057858,
            -0.4338788,
            0.84871036,
            0.69200915,
            -0.31601277,
            -2.1152194,
            0.32227492,
            -1.2633348,
            0.3499832,
            0.30813393,
            0.11984151,
            1.2376579,
            1.1167772,
            -0.24727815,
            -1.3526537,
            -1.6959312,
            0.5666506,
            0.79350835,
            0.59883946,
            -1.5550951,
            -0.3413604,
            1.8530061,
            0.7501895,
            -0.58549756,
            -0.17339675,
            0.18347794,
            1.3893661,
            1.5863342,
            0.94629836,
            -0.84367675,
            -0.6135831,
            0.03159274,
            -0.49267697,
            0.24841475,
            0.43969584,
            0.112411186,
            0.64079237,
            0.44115627,
            -0.10230965,
            0.792444,
            -0.2896677,
            0.052507486,
            0.52286047,
            2.3022053,
            -1.4688939,
            -1.5866888,
            -0.6730899,
            0.8728312,
            1.0553575,
            0.17784372,
            -0.23033547,
            -0.3917544,
            0.5432947,
            -0.39515755,
            -0.44621718,
            0.7440207,
            1.5209795,
            2.3803675,
            -1.1256016,
            -0.3169981,
            -1.0924683,
            -0.0851943,
            -0.093348235,
            0.6870502,
            -0.83831537,
            0.018486667,
            -0.7504268,
            0.18540798,
            0.62113833,
            0.63818157,
            -0.24600095,
            2.3025165,
            -1.8816892,
        ];
        let input = from_vec::<f32, _, D>(input, [1, 3, 5, 5]);
        let output = max_pool_2d(
            input.clone(),
            (3, 3),
            (2, 2),
            (1, 1),
            MaxPool2dConfig::default(),
        );
        let forward_ans = vec![
            0.69200915, 0.32227492, 0.84871036, 0.69200915, 1.2376579, 1.2376579, 0.59883946,
            1.8530061, 1.8530061, 0.94629836, 1.3893661, 1.5863342, 0.94629836, 0.792444,
            0.64079237, 2.3022053, 0.792444, 0.8728312, 1.0553575, 1.5209795, 2.3803675, 0.6870502,
            1.5209795, 2.3803675, 0.6870502, 2.3025165, 2.3025165,
        ];
        let forward_ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(forward_ans, [1, 3, 3, 3]);
        assert_val_eq!(output.clone(), forward_ans, 1e-6);
        let grad_ans = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
        ];
        let grad_ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(grad_ans, [1, 3, 5, 5]);
        output.backward();
        assert_val_eq_grad!(input, grad_ans, 1e-6);
    }
    run_test!(_pool2d, pool_2d_cpu, pool_2d_nvidia);
}
