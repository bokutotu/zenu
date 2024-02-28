use std::{cell::RefCell, rc::Rc};

use ruml_matrix::{
    dim::{DimDyn, DimTrait},
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory_impl::OwnedMem,
    num::Num,
    operation::{mul::Gemm, transpose::Transpose, zeros::Zeros},
};

use crate::{Function, Variable, VariableWeak};

use super::output_shape;

struct MatMul<T: Num> {
    x: Variable<T>,
    y: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num> MatMul<T> {
    pub fn new(x: Variable<T>, y: Variable<T>, output: Variable<T>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl<T: Num> Function<T> for MatMul<T> {
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
        let x = self.x.get_data();
        let y = self.y.get_data();
        let x_shape = x.shape();
        let y_shape = y.shape();
        let mut x_grad: Matrix<OwnedMem<T>, DimDyn> = Zeros::zeros(x_shape);
        let mut y_grad: Matrix<OwnedMem<T>, DimDyn> = Zeros::zeros(y_shape);
        self.output.upgrade().unwrap().with_grad_data(|grad| {
            let mut y_t = y.to_view();
            y_t.transpose();
            let mut x_t = x.to_view();
            x_t.transpose();
            x_grad.to_view_mut().gemm(grad.to_view(), y_t);
            y_grad.to_view_mut().gemm(x_t, grad.to_view());
        });
        *self.x.get_grad_mut() = Some(Variable::new(x_grad));
        *self.y.get_grad_mut() = Some(Variable::new(y_grad));
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

pub fn matmul<T: Num>(x: Variable<T>, y: Variable<T>) -> Variable<T> {
    let output_shape = DimDyn::new(&[x.get_data().shape()[0], y.get_data().shape()[1]]);
    println!("output_shape: {:?}", output_shape);

    let output = Zeros::zeros(output_shape);
    let output = Variable::new(output);
    let matmul = MatMul::new(x, y, output.clone());
    matmul.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(matmul))));
    output
}

#[cfg(test)]
mod matmul {
    use ruml_matrix::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn};

    use crate::Variable;

    use super::matmul;

    #[test]
    fn matmul_test() {
        let x = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let y = OwnedMatrixDyn::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2]);

        let x = Variable::new(x);
        let y = Variable::new(y);

        let output = matmul(x, y);
        output.backward();
    }
}
