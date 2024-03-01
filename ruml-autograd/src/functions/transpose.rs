use std::{cell::RefCell, rc::Rc};

use ruml_matrix::{
    dim::{DimDyn, DimTrait},
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory_impl::OwnedMem,
    num::Num,
    operation::{copy_from::CopyFrom, transpose::Transpose as T, zeros::Zeros},
};

use crate::{Function, Variable, VariableWeak};

struct Transpose<T: Num> {
    x: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num> Transpose<T> {
    pub fn new(x: Variable<T>, output: Variable<T>) -> Self {
        let output = output.downgrade();
        Self { x, output }
    }
}

impl<T: Num> Function<T> for Transpose<T> {
    // FIXME: メモリを使いまわす
    fn forward(&self) {
        let x = self.x.get_data();
        let mut out: Matrix<OwnedMem<T>, DimDyn> = Zeros::zeros(x.shape());
        out.to_view_mut().copy_from(&x.to_view());
        out.transpose();
        let output = self.output.upgrade().unwrap();
        output
            .get_data_mut()
            .to_view_mut()
            .copy_from(&out.to_view());
        println!("out");
        println!("{:?}", out.to_view());
    }

    // FIXME: メモリを使いまわす
    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let grad = output.get_grad().clone().unwrap();
        self.x.set_grad(transpose(grad));
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.x.clone()]
    }
}

pub fn transpose<T: Num>(x: Variable<T>) -> Variable<T> {
    if x.get_data().shape().len() < 2 {
        panic!("Not implemented yet");
    }
    let output_shape = x.get_data().shape_stride().transpose().shape();
    let output = Zeros::zeros(output_shape);
    let output = Variable::new(output);
    let transpose = Transpose::new(x, output.clone());
    transpose.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(transpose))));
    output
}

#[cfg(test)]
mod transpose {
    use ruml_matrix::{
        dim::DimDyn,
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::Matrix,
        memory_impl::OwnedMem,
        operation::asum::Asum,
    };

    use crate::Variable;

    use super::transpose;

    #[test]
    fn test_transpose() {
        let x: Matrix<OwnedMem<f32>, DimDyn> =
            Matrix::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let x = Variable::from(x);
        let y = transpose(x);
        let y = y.get_data();
        println!("{:?}", y.to_view());
        let ans: Matrix<OwnedMem<f32>, DimDyn> =
            OwnedMatrix::from_vec(vec![1., 4., 2., 5., 3., 6.], [3, 2]);
        let diff = y.to_view() - ans.to_view();
        assert!(diff.asum() < 1e-6);
    }
}
