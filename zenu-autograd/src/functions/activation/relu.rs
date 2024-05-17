use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    constructor::zeros::Zeros,
    dim::DimDyn,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::{Matrix, OwnedMatrixDyn},
    memory_impl::OwnedMem,
    num::Num,
    operation::relu::Relu as R,
};

use crate::{Function, Variable, VariableWeak};

struct Relu<T: Num, D: Device> {
    input: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Relu<T, D> {
    pub fn new(input: Variable<T, D>, output: Variable<T, D>) -> Self {
        let output = output.downgrade();
        Self { input, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for Relu<T, D> {
    fn forward(&self) {
        let input = self.input.get_data();
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        let mut output_view_mut = output.to_ref_mut();
        R::relu(&mut output_view_mut, input.to_ref());
    }

    fn backward(&self) {
        // リファレンスカウンタの関係でスコープを切る必要がある
        let input_grad = {
            let input = self.input.get_data();
            let output = self.output.upgrade().unwrap();
            let output_grad = output.get_grad().clone().unwrap();
            let mut mask = OwnedMatrixDyn::zeros(input.shape());
            R::relu_backward_mask(&mut mask.to_ref_mut(), input.to_ref());
            let mask = Variable::from(mask);
            output_grad * mask
        };
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

pub fn relu<T: Num, D: Device>(input: Variable<T, D>) -> Variable<T, D> {
    let output: Matrix<OwnedMem<T, D>, DimDyn> = Zeros::zeros(input.get_data().shape());
    let output = Variable::from(output);
    let relu = Relu::new(input, output.clone());
    relu.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(relu))));
    output
}

#[cfg(test)]
mod relu {
    use zenu_matrix::{
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::asum::Asum,
    };

    use crate::Variable;

    use super::relu;

    #[test]
    fn relu_1d() {
        let x = OwnedMatrixDyn::from_vec(vec![-1., 0., 2., 3.], [2, 2]);
        let x_v = Variable::from(x.clone());
        let y = relu(x_v.clone());
        let ans = OwnedMatrixDyn::from_vec(vec![0., 0., 2., 3.], [2, 2]);
        let diff = y.get_data().to_ref() - ans.to_ref();
        let diff_asum = diff.asum();
        assert!(diff_asum < 1.0e-6);

        y.backward();

        let ans = OwnedMatrixDyn::from_vec(vec![0., 0., 1., 1.], [2, 2]);

        x_v.with_grad_data(|x_grad| {
            let diff = x_grad.to_ref() - ans.to_ref();
            let diff_asum = diff.asum();
            assert!(diff_asum < 1.0e-6);
        });
    }
}
