use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{device::Device, num::Num};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

use super::sum::sum;

struct SoftMax<T: Num, D: Device> {
    input: Variable<T, D>,
    output: VariableWeak<T, D>,
    axis: usize,
}

impl<T: Num, D: Device> SoftMax<T, D> {
    fn new(input: Variable<T, D>, output: VariableWeak<T, D>, axis: usize) -> Self {
        Self {
            input,
            output,
            axis,
        }
    }
}

impl<T: Num, D: Device> Function<T, D> for SoftMax<T, D> {
    fn forward(&self) {
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        output
            .to_ref_mut()
            .softmax_assign(&self.input.get_data().to_ref(), self.axis);
    }

    fn backward(&self) {
        let y = self.output.upgrade().unwrap();
        let y_grad = y.get_grad().unwrap();
        let gx = y.clone() * y_grad;
        let sum_gx = sum(gx.clone(), self.axis, true);
        let gx = gx - (y * sum_gx);
        self.input.set_grad(gx);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

pub fn softmax<T: Num, D: Device>(input: Variable<T, D>, axis: usize) -> Variable<T, D> {
    let output = alloc(input.get_shape());
    let softmax = SoftMax::new(input, output.clone().downgrade(), axis);
    softmax.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(softmax))));
    output
}

#[cfg(test)]
mod softmax {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::creator::from_vec::from_vec;
    use crate::Variable;

    use super::softmax;

    fn softmax_2d_1d<D: Device>() {
        let input: Matrix<Owned<f64>, DimDyn, D> =
            Matrix::from_vec(vec![1.0, 2.0, 3.0, 4., 5., 6., 7., 8.], [2, 4]);
        let input = Variable::from(input);
        let output = softmax(input.clone(), 1);

        let expected: Matrix<Owned<f64>, DimDyn, D> = Matrix::from_vec(
            vec![
                0.0320586, 0.08714432, 0.23688284, 0.64391428, 0.0320586, 0.08714432, 0.23688284,
                0.64391428,
            ],
            [2, 4],
        );
        assert_val_eq!(output.clone(), expected, 1e-6);

        let tmp = from_vec(vec![1.0, 2.0, 3.0, 4.0], [4]);
        let output = output * tmp;
        output.backward();
        let input_grad: Matrix<Owned<f64>, DimDyn, D> = Matrix::from_vec(
            vec![
                -0.07991097,
                -0.13007621,
                -0.11670097,
                0.32668814,
                -0.07991097,
                -0.13007621,
                -0.11670097,
                0.32668814,
            ],
            [2, 4],
        );
        assert_val_eq_grad!(input, input_grad, 1e-6);
    }
    run_test!(softmax_2d_1d, softmax_2d_1d_cpu, softmax_2d_1d_nvidia);
}
