use std::{cell::RefCell, ops::Mul, rc::Rc};

use zenu_matrix::{device::Device, num::Num};

use crate::{creator::zeros::zeros, Function, Variable, VariableWeak};

use super::{output_shape, sum_to::sum_to};

struct Multiply<T: Num, D: Device> {
    x: Variable<T, D>,
    y: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Multiply<T, D> {
    pub fn new(x: Variable<T, D>, y: Variable<T, D>, output: Variable<T, D>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for Multiply<T, D> {
    fn forward(&self) {
        let x = self.x.get_data();
        let y = self.y.get_data();
        let x = x.to_ref();
        let y = y.to_ref();
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        output.to_ref_mut().mul_array(&x, &y);
    }

    fn backward(&self) {
        let x_shape = self.x.get_data().shape();
        let y_shape = self.y.get_data().shape();
        let output = self.output.upgrade().unwrap();
        let grad = output.get_grad().clone().unwrap();
        let x_grad = grad.clone() * self.y.clone();
        let y_grad = self.x.clone() * grad;
        self.x.set_grad(sum_to(x_grad, x_shape));
        self.y.set_grad(sum_to(y_grad, y_shape));
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

fn mul<T: Num, D: Device>(x: Variable<T, D>, y: Variable<T, D>) -> Variable<T, D> {
    let output_shape = output_shape(&x, &y);
    let output = zeros(output_shape);
    let mul = Multiply::new(x, y, output.clone());
    mul.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(mul))));
    output
}

impl<T: Num, D: Device> Mul<Variable<T, D>> for Variable<T, D> {
    type Output = Variable<T, D>;

    fn mul(self, rhs: Variable<T, D>) -> Self::Output {
        mul(self, rhs)
    }
}

#[cfg(test)]
mod mul {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::Variable;

    fn mul_2d_1d<D: Device>() {
        let a_mat: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let b_mat: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1., 2., 3.], [3]);
        let a = Variable::new(a_mat);
        let b = Variable::new(b_mat);
        let c = a.clone() * b.clone();
        c.backward();
        let c_ans: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1., 4., 9., 4., 10., 18.], [2, 3]);
        let a_grad_ans: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1., 2., 3., 1., 2., 3.], [2, 3]);
        let b_grad_ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![5., 7., 9.], [3]);
        assert_val_eq!(c, c_ans, 1e-6);
        assert_val_eq_grad!(a, a_grad_ans, 1e-6);
        assert_val_eq_grad!(b, b_grad_ans, 1e-6);
    }
    run_test!(mul_2d_1d, mul_2d_1d_cpu, mul_2d_1d_gpu);
}
