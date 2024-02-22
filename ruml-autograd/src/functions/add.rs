use std::{cell::RefCell, ops::Add, rc::Rc};

use ruml_matrix::{
    dim::DimDyn,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory::Owned,
    memory_impl::OwnedMem,
    operation::{add::MatrixAdd, zeros::Zeros},
};

use crate::{Function, Variable, VariableWeak};

use super::{gradient_sum_over_axis, output_shape};

struct Addition<M: Owned> {
    x: Variable<M>,
    y: Variable<M>,
    output: VariableWeak<M>,
}

impl<M: Owned> Addition<M> {
    pub fn new(x: Variable<M>, y: Variable<M>, output: Variable<M>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl Function<OwnedMem<f32>> for Addition<OwnedMem<f32>> {
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
        let mut x_grad: Matrix<OwnedMem<f32>, DimDyn> = Zeros::zeros(x_shape);
        let mut y_grad: Matrix<OwnedMem<f32>, DimDyn> = Zeros::zeros(y_shape);
        self.output.upgrade().unwrap().with_grad_data(|grad| {
            gradient_sum_over_axis(grad.to_view(), x_grad.to_view_mut());
            gradient_sum_over_axis(grad.to_view(), y_grad.to_view_mut());
        });
        *self.x.get_grad_mut() = Some(Variable::new(x_grad));
        *self.y.get_grad_mut() = Some(Variable::new(y_grad));
    }

    fn get_inputs(&self) -> Vec<Variable<OwnedMem<f32>>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

fn add(x: Variable<OwnedMem<f32>>, y: Variable<OwnedMem<f32>>) -> Variable<OwnedMem<f32>> {
    let output_shape = output_shape(&x, &y);
    let output = Zeros::zeros(output_shape);
    let output = Variable::new(output);
    let add = Addition::new(x, y, output.clone());
    add.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(add))));
    output
}

impl Add<Variable<OwnedMem<f32>>> for Variable<OwnedMem<f32>> {
    type Output = Variable<OwnedMem<f32>>;

    fn add(self, other: Variable<OwnedMem<f32>>) -> Self::Output {
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
