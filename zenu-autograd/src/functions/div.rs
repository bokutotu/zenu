use std::{cell::RefCell, ops::Div, rc::Rc};

use zenu_matrix::{device::Device, num::Num};

use crate::{creator::zeros::zeros, Function, Variable, VariableWeak};

struct DivFunc<T: Num, D: Device> {
    x: Variable<T, D>,
    y: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> DivFunc<T, D> {
    pub fn new(x: Variable<T, D>, y: Variable<T, D>, output: Variable<T, D>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for DivFunc<T, D> {
    fn forward(&self) {
        let x = self.x.get_data();
        let y = self.y.get_data();
        let output = x.to_ref() / y.to_ref();
        self.output
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&output.to_ref());
    }

    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().clone().unwrap();
        let x_grad = output_grad.clone() / self.y.clone();
        let y_grad = output_grad * self.x.clone() / (self.y.clone() * self.y.clone())
            * Variable::from(T::minus_one());
        self.x.set_grad(x_grad);
        self.y.set_grad(y_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

pub fn div<T: Num, D: Device>(x: Variable<T, D>, y: Variable<T, D>) -> Variable<T, D> {
    let output_shape = super::output_shape(&x, &y);
    let output = zeros(output_shape);
    let div = DivFunc::new(x, y, output.clone());
    div.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(div))));
    output
}

impl<T: Num, D: Device> Div<Variable<T, D>> for Variable<T, D> {
    type Output = Variable<T, D>;

    fn div(self, rhs: Variable<T, D>) -> Self::Output {
        div(self, rhs)
    }
}

#[cfg(test)]
mod div {

    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::{creator::from_vec::from_vec, Variable};

    fn div_2d<D: Device>() {
        let a: Variable<f64, D> = from_vec(vec![1f64, 2., 3., 4., 5., 6.], [2, 3]);
        let b: Variable<f64, D> = from_vec(vec![6., 7., 8., 9., 10., 11.], [2, 3]);
        let c = a.clone() / b.clone();
        c.backward();
        let ans = Matrix::<Owned<f64>, DimDyn, D>::from_vec(
            vec![1. / 6., 2. / 7., 3. / 8., 4. / 9., 5. / 10., 6. / 11.],
            [2, 3],
        );
        assert_val_eq!(c, ans, 1e-6);
        let a_grad: Matrix<Owned<f64>, DimDyn, D> =
            Matrix::from_vec(vec![0.1667, 0.1429, 0.1250, 0.1111, 0.1000, 0.0909], [2, 3]);
        let b_grad: Matrix<Owned<f64>, DimDyn, D> = Matrix::from_vec(
            vec![-0.0278, -0.0408, -0.0469, -0.0494, -0.0500, -0.0496],
            [2, 3],
        );
        assert_val_eq_grad!(a, a_grad, 1e-4);
        assert_val_eq_grad!(b, b_grad, 1e-4);
    }
    run_test!(div_2d, div_2d_cpu, div_2d_nvidia);
}
