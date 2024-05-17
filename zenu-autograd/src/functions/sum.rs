use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    constructor::zeros::Zeros,
    dim::LessDimTrait,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::OwnedMatrixDyn,
    num::Num,
    operation::{add_axis::MatrixAddAxis, copy_from::CopyFrom, sum::MatrixSum},
};

use crate::{Function, Variable, VariableWeak};

use super::broadcast::broadcast;

struct Sum<T: Num> {
    input: Variable<T>,
    output: VariableWeak<T>,
    axis: usize,
    keep_dim: bool,
}

impl<T: Num, D: Device> Sum<T> {
    pub fn new(input: Variable<T>, output: VariableWeak<T>, axis: usize, keep_dim: bool) -> Self {
        Self {
            input,
            output,
            axis,
            keep_dim,
        }
    }
}

impl<T: Num, D: Device> Function<T> for Sum<T> {
    fn forward(&self) {
        let input = self.input.get_data();
        let input = input.to_view();
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        let mut output = output.to_view_mut();
        let ans = input.sum(self.axis, self.keep_dim);
        output.copy_from(&ans.to_view());
    }

    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().clone().unwrap();
        let input_grad = broadcast(output_grad, self.input.get_data().shape());
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.input.clone()]
    }
}

// FIXME: 汚いのでどうにかする
pub fn sum<T: Num>(input: Variable<T>, axis: usize, keep_dim: bool) -> Variable<T> {
    let output_shape = input.get_data().shape().remove_axis(axis);
    let mut zeros = OwnedMatrixDyn::zeros(output_shape);
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
        dim::DimTrait,
        matrix::{MatrixBase, OwnedMatrix, ToViewMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::asum::Asum,
    };

    use crate::creator::from_vec::from_vec;

    use super::sum;

    #[test]
    fn sum_2d_1d_keep_dim() {
        let input = from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let output = sum(input, 1, true);
        let ans = OwnedMatrixDyn::from_vec(vec![6., 15.], [2, 1]);
        assert_eq!((output.get_data().to_view() - ans.to_view()).asum(), 0.);
    }

    #[test]
    fn sum_3d_keep_dim() {
        let input = from_vec(
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24., 25., 26., 27.,
            ],
            [3, 3, 3],
        );
        let output = sum(input, 0, true);
        assert_eq!(output.get_data().shape().slice(), [1, 3, 3]);
        output.backward();
        let ans =
            OwnedMatrixDyn::from_vec(vec![30., 33., 36., 39., 42., 45., 48., 51., 54.], [1, 3, 3]);
        let diff = output.get_data().to_view() - ans;
        let diff = diff.asum();
        assert!(diff < 1e-6);
    }
}
