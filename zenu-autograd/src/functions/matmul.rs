use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    dim::{DimDyn, DimTrait},
    num::Num,
    operation::mul::gemm_assign,
};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

use super::transpose::transpose;

struct MatMul<T: Num, D: Device> {
    x: Variable<T, D>,
    y: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> MatMul<T, D> {
    pub fn new(x: Variable<T, D>, y: Variable<T, D>, output: Variable<T, D>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for MatMul<T, D> {
    fn forward(&self) {
        // if self.x.get_data().shape().len() != 2 || self.y.get_data().shape().len() != 2 {
        //     panic!("x.shape().len() != 2 || y.shape().len() != 2");
        // }
        assert_eq!(
            self.x.get_shape().len() == 2,
            self.y.get_shape().len() == 2,
            "x.shape().len() != y.shape().len()"
        );

        let x = self.x.get_data();
        let y = self.y.get_data();
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        let x = x.to_ref();
        let y = y.to_ref();
        let output = output.to_ref_mut();
        gemm_assign(&x, &y, &output, T::one(), T::zero());
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let grad = output.get_grad().clone().unwrap();
        let x_grad = matmul(grad.clone(), transpose(self.y.clone()));
        let y_grad = matmul(transpose(self.x.clone()), grad);
        self.x.set_grad(x_grad);
        self.y.set_grad(y_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

#[must_use]
pub fn matmul<T: Num, D: Device>(x: Variable<T, D>, y: Variable<T, D>) -> Variable<T, D> {
    let output_shape = DimDyn::new(&[x.get_data().shape()[0], y.get_data().shape()[1]]);
    let output = alloc(output_shape);
    let matmul = MatMul::new(x, y, output.clone());
    matmul.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(matmul))));
    output
}

#[cfg(test)]
mod matmul {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::Variable;

    use super::matmul;

    fn matmul_test<D: Device>() {
        let x = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let x = Matrix::<Owned<f64>, DimDyn, D>::from_vec(x, [3, 4]);
        let y = Matrix::<Owned<f64>, DimDyn, D>::from_vec(y, [4, 2]);

        let x = Variable::new(x);
        let y = Variable::new(y);

        let output = matmul(x.clone(), y.clone());
        output.backward();
        let ans = vec![50., 60., 114., 140., 178., 220.];
        let ans = Matrix::<Owned<f64>, DimDyn, D>::from_vec(ans, [3, 2]);
        assert_val_eq!(output, ans, 1e-6);
        let x_grad_ans = vec![3., 7., 11., 15., 3., 7., 11., 15., 3., 7., 11., 15.];
        let x_grad_ans = Matrix::<Owned<f64>, DimDyn, D>::from_vec(x_grad_ans, [3, 4]);
        assert_val_eq_grad!(x, x_grad_ans, 1e-6);
        let y_grad_ans = vec![15., 15., 18., 18., 21., 21., 24., 24.];
        let y_grad_ans = Matrix::<Owned<f64>, DimDyn, D>::from_vec(y_grad_ans, [4, 2]);
        assert_val_eq_grad!(y, y_grad_ans, 1e-6);
    }
    run_test!(matmul_test, matmul_cpu, matmul_nvidia);
}
