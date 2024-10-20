use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{device::Device, num::Num};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

struct Clip<T: Num, D: Device> {
    min: T,
    max: T,
    input: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Clip<T, D> {
    pub fn new(min: T, max: T, input: Variable<T, D>, output: Variable<T, D>) -> Self {
        assert_eq!(
            input.get_data().shape(),
            output.get_data().shape(),
            "input.shape() != output.shape()"
        );
        let output = output.downgrade();
        Self {
            min,
            max,
            input,
            output,
        }
    }
}

impl<T: Num, D: Device> Function<T, D> for Clip<T, D> {
    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().clone().unwrap();
        let clip_filter = self
            .input
            .get_data()
            .to_ref()
            .clip_backward_mask(self.min, self.max);
        let clip_filter = Variable::from(clip_filter);
        let input_grad = output_grad * clip_filter;
        self.input.set_grad(input_grad);
    }

    fn forward(&self) {
        let input = self.input.get_data();
        let output = input.clip(self.min, self.max);
        self.output
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&output.to_ref());
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

pub fn clip<T: Num, D: Device>(input: Variable<T, D>, min: T, max: T) -> Variable<T, D> {
    let output = alloc(input.get_shape());
    let clip = Clip::new(min, max, input, output.clone());
    clip.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(clip))));
    output
}

#[cfg(test)]
mod clip {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::creator::from_vec::from_vec;

    fn clip_1d<D: Device>() {
        let input = from_vec(vec![1., 2., 3., 4., 5., 6.], [6]);
        let output = super::clip(input.clone(), 2.0, 4.0);
        output.backward();
        let ans: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![2., 2., 3., 4., 4., 4.], [6]);
        let grad_ans: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![0., 1., 1., 1., 0., 0.], [6]);
        assert_val_eq!(output, ans, 1.0e-6);
        assert_val_eq_grad!(input, grad_ans, 1.0e-6);
    }
    run_test!(clip_1d, clip_1d_cpu, clip_1d_nvidia);
}
