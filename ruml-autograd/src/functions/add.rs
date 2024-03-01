use std::{cell::RefCell, ops::Add, rc::Rc};

use ruml_matrix::{
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    num::Num,
    operation::{add::MatrixAdd, zeros::Zeros},
};

use crate::{Function, Variable, VariableWeak};

use super::{output_shape, sum_to::sum_to};

struct Addition<T: Num> {
    x: Variable<T>,
    y: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num> Addition<T> {
    pub fn new(x: Variable<T>, y: Variable<T>, output: Variable<T>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl<T: Num> Function<T> for Addition<T> {
    fn forward(&self) {
        let x = self.x.get_data();
        let y = self.y.get_data();
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        MatrixAdd::add(output.to_view_mut(), x.to_view(), y.to_view());
    }

    fn backward(&self) {
        let x_shape = self.x.get_data().shape();
        let y_shape = self.y.get_data().shape();
        let output = self.output.upgrade().unwrap();
        let grad = output.get_grad().clone().unwrap();
        self.x.set_grad(sum_to(grad.clone(), x_shape));
        self.y.set_grad(sum_to(grad, y_shape));
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

fn add<T: Num>(x: Variable<T>, y: Variable<T>) -> Variable<T> {
    let output_shape = output_shape(&x, &y);
    let output = Zeros::zeros(output_shape);
    let output = Variable::new(output);
    let add = Addition::new(x, y, output.clone());
    add.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(add))));
    output
}

impl<T: Num> Add<Variable<T>> for Variable<T> {
    type Output = Variable<T>;

    fn add(self, other: Variable<T>) -> Self::Output {
        add(self, other)
    }
}

#[cfg(test)]
mod add {
    use ruml_matrix::{
        matrix::ToViewMatrix,
        matrix_impl::OwnedMatrixDyn,
        operation::{asum::Asum, ones::Ones},
    };

    use crate::Variable;

    #[test]
    fn add() {
        let x: OwnedMatrixDyn<f32> = Ones::ones([100, 200]);
        let y: OwnedMatrixDyn<f32> = Ones::ones([20, 100, 200]);
        let x_val = Variable::new(x);
        let y_val = Variable::new(y);
        let z = x_val.clone() + y_val.clone();
        z.backward();
        let z_data = z.get_data();
        let ans: OwnedMatrixDyn<f32> = OwnedMatrixDyn::ones([20, 100, 200]).to_view() * 2.0;
        let diff = z_data.to_view() - ans.to_view();
        let diff_sum = diff.asum();
        assert!(diff_sum < 1e-6);

        x_val.with_grad_data(|grad| {
            let grad = grad.to_view();
            let ans: OwnedMatrixDyn<f32> = OwnedMatrixDyn::ones([100, 200]).to_view() * 20.;
            let diff = grad - ans.to_view();
            let diff_sum = diff.asum();
            assert!(diff_sum < 1e-6);
        });
        y_val.with_grad_data(|grad| {
            let grad = grad.to_view();
            let ans: OwnedMatrixDyn<f32> = OwnedMatrixDyn::ones([20, 100, 200]);
            let diff = grad - ans.to_view();
            let diff_sum = diff.asum();
            assert!(diff_sum < 1e-6);
        });
    }
}
