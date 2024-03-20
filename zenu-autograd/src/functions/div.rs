use std::{cell::RefCell, ops::Div, rc::Rc};

use zenu_matrix::{
    matrix::{ToViewMatrix, ToViewMutMatrix},
    num::Num,
    operation::copy_from::CopyFrom,
};

use crate::{Function, Variable, VariableWeak};

struct DivFunc<T: Num> {
    x: Variable<T>,
    y: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num> DivFunc<T> {
    pub fn new(x: Variable<T>, y: Variable<T>, output: Variable<T>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl<T: Num> Function<T> for DivFunc<T> {
    fn forward(&self) {
        let x = self.x.get_data();
        let y = self.y.get_data();
        let output = x / y;
        self.output
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_view_mut()
            .copy_from(&output.to_view());
    }

    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().clone().unwrap();
        let x_grad = output_grad.clone() / self.y.clone();
        let y_grad = output_grad * self.x.clone() / (self.y.clone() * self.y.clone())
            * Variable::from(T::minus_one());
        self.x.set_grad(x_grad);
        self.y.set_grad(y_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

pub fn div<T: Num>(x: Variable<T>, y: Variable<T>) -> Variable<T> {
    let output = Variable::new(x.get_data() / y.get_data());
    let div = DivFunc::new(x, y, output.clone());
    div.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(div))));
    output
}

impl<T: Num> Div<Variable<T>> for Variable<T> {
    type Output = Variable<T>;

    fn div(self, rhs: Variable<T>) -> Self::Output {
        div(self, rhs)
    }
}

#[cfg(test)]
mod div {
    use zenu_matrix::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};

    use crate::creator::from_vec::from_vec;

    #[test]
    fn div_2d() {
        let a = from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let b = from_vec(vec![6., 7., 8., 9., 10., 11.], [2, 3]);
        let c = a.clone() / b.clone();
        c.backward();
        let ans = OwnedMatrixDyn::from_vec(
            vec![1. / 6., 2. / 7., 3. / 8., 4. / 9., 5. / 10., 6. / 11.],
            [2, 3],
        );
        let diff = c.get_data() - ans;
        let diff_asum = diff.asum();
        assert_eq!(diff_asum, 0.0);

        let x_grad_ans =
            OwnedMatrixDyn::from_vec(vec![0.1667, 0.1429, 0.1250, 0.1111, 0.1000, 0.0909], [2, 3]);
        let y_grad_ans = OwnedMatrixDyn::from_vec(
            vec![-0.0278, -0.0408, -0.0469, -0.0494, -0.0500, -0.0496],
            [2, 3],
        );
        let x_grad = a.get_grad().unwrap().get_data();
        let y_grad = b.get_grad().unwrap().get_data();
        let diff_x = x_grad - x_grad_ans;
        let diff_y = y_grad - y_grad_ans;
        let diff_x_asum = diff_x.asum();
        let diff_y_asum = diff_y.asum();
        assert!(diff_x_asum < 1e-4);
        assert!(diff_y_asum < 1e-4);
    }
}
