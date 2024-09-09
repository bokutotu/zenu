use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    dim::{DimDyn, LessDimTrait},
    matrix::{Matrix, Owned},
    num::Num,
};

use crate::{Function, Variable, VariableWeak};

use super::broadcast::broadcast;

struct Sum<T: Num, D: Device> {
    input: Variable<T, D>,
    output: VariableWeak<T, D>,
    axis: usize,
    keep_dim: bool,
}

impl<T: Num, D: Device> Sum<T, D> {
    pub fn new(
        input: Variable<T, D>,
        output: VariableWeak<T, D>,
        axis: usize,
        keep_dim: bool,
    ) -> Self {
        Self {
            input,
            output,
            axis,
            keep_dim,
        }
    }
}

impl<T: Num, D: Device> Function<T, D> for Sum<T, D> {
    fn forward(&self) {
        let input = self.input.get_data();
        let input = input.to_ref();
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        let output = output.to_ref_mut();
        let ans = input.sum(self.axis, self.keep_dim);
        output.copy_from(&ans.to_ref());
    }

    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().clone().unwrap();
        let input_grad = broadcast(output_grad, self.input.get_data().shape());
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

// FIXME: 汚いのでどうにかする
#[must_use]
pub fn sum<T: Num, D: Device>(
    input: Variable<T, D>,
    axis: usize,
    keep_dim: bool,
) -> Variable<T, D> {
    let output_shape = input.get_data().shape().remove_axis(axis);
    let mut zeros: Matrix<Owned<T>, DimDyn, D> = Matrix::zeros(output_shape);
    if keep_dim {
        zeros.add_axis(axis);
    }
    let output = Variable::from(zeros);
    let sum = Sum::new(input, output.clone().downgrade(), axis, keep_dim);
    sum.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(sum))));
    output
}

#[cfg(test)]
mod sum {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, run_test};

    use crate::creator::from_vec::from_vec;

    use super::sum;

    fn sum_2d_1d_keep_dim<D: Device>() {
        let input = from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let output = sum(input, 1, true);
        let ans: Matrix<Owned<f64>, DimDyn, D> = Matrix::from_vec(vec![6., 15.], [2, 1]);
        assert_val_eq!(output, ans, 0.);
    }
    run_test!(
        sum_2d_1d_keep_dim,
        sum_2d_1d_keep_dim_cpu,
        sum_2d_1d_keep_dim_nvidia
    );

    fn sum_3d_keep_dim<D: Device>() {
        let input = from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24., 25., 26., 27.,
            ],
            [3, 3, 3],
        );
        let output = sum(input, 0, true);
        output.backward();
        let ans: Matrix<Owned<f64>, DimDyn, D> =
            Matrix::from_vec(vec![30., 33., 36., 39., 42., 45., 48., 51., 54.], [1, 3, 3]);
        assert_val_eq!(output, ans, 0.);
    }
    run_test!(sum_3d_keep_dim, sum_3d_keep_dim_cpu, sum_3d_keep_dim_nvidia);
}
