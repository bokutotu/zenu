use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    constructor::zeros::Zeros,
    dim::{DimDyn, DimTrait},
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    num::Num,
    operation::mul::Gemm,
};

use crate::{Function, Variable, VariableWeak};

use super::transpose::transpose;

struct MatMul<T: Num> {
    x: Variable<T>,
    y: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num, D: Device> MatMul<T> {
    pub fn new(x: Variable<T>, y: Variable<T>, output: Variable<T>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl<T: Num, D: Device> Function<T> for MatMul<T> {
    fn forward(&self) {
        if self.x.get_data().shape().len() != 2 || self.y.get_data().shape().len() != 2 {
            panic!("x.shape().len() != 2 || y.shape().len() != 2");
        }

        let x = self.x.get_data();
        let y = self.y.get_data();
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        let x = x.to_view();
        let y = y.to_view();
        let output = output.to_view_mut();
        output.gemm(x, y);
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let grad = output.get_grad().clone().unwrap();
        let x_grad = matmul(grad.clone(), transpose(self.y.clone()));
        let y_grad = matmul(transpose(self.x.clone()), grad);
        self.x.set_grad(x_grad);
        self.y.set_grad(y_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

pub fn matmul<T: Num>(x: Variable<T>, y: Variable<T>) -> Variable<T> {
    let output_shape = DimDyn::new(&[x.get_data().shape()[0], y.get_data().shape()[1]]);
    let output = Zeros::zeros(output_shape);
    let output = Variable::new(output);
    let matmul = MatMul::new(x, y, output.clone());
    matmul.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(matmul))));
    output
}

#[cfg(test)]
mod matmul {
    use zenu_matrix::{
        dim::DimDyn,
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::Matrix,
        memory_impl::OwnedMem,
        operation::asum::Asum,
    };

    use crate::Variable;

    use super::matmul;

    #[test]
    fn matmul_test() {
        let x = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let x = OwnedMatrix::from_vec(x, &[3, 4]);
        let y = OwnedMatrix::from_vec(y, &[4, 2]);

        let x = Variable::new(x);
        let y = Variable::new(y);

        let output = matmul(x.clone(), y.clone());
        let ans = vec![50., 60., 114., 140., 178., 220.];
        let ans: Matrix<OwnedMem<f64>, DimDyn> = OwnedMatrix::from_vec(ans, &[3, 2]);
        let diff = output.get_data().to_view() - ans.to_view();
        let diff_asum = diff.asum();
        assert!(diff_asum < 1e-6);

        output.backward();
        x.with_grad_data(|grad| {
            let ans = vec![3., 7., 11., 15., 3., 7., 11., 15., 3., 7., 11., 15.];
            let ans: Matrix<OwnedMem<f64>, DimDyn> = OwnedMatrix::from_vec(ans, &[3, 4]);
            let diff = grad.to_view() - ans.to_view();
            let diff_asum = diff.asum();
            assert!(diff_asum < 1e-6);
        });
        y.with_grad_data(|grad| {
            let ans = vec![15., 15., 18., 18., 21., 21., 24., 24.];
            let ans: Matrix<OwnedMem<f64>, DimDyn> = OwnedMatrix::from_vec(ans, &[4, 2]);
            let diff = grad.to_view() - ans.to_view();
            let diff_asum = diff.asum();
            assert!(diff_asum < 1e-6);
        });
    }
}
