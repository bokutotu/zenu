use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    dim::DimDyn,
    matrix::{Matrix, Owned},
    num::Num,
};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

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
        let output_view_mut = output.to_ref_mut();
        output_view_mut.relu(&input.to_ref(), T::zero());
    }

    fn backward(&self) {
        // リファレンスカウンタの関係でスコープを切る必要がある
        // TODO: 複数回微分の場合に対応する
        let input_grad = {
            let input = self.input.get_data();
            let output = self.output.upgrade().unwrap();
            let output_grad = output.get_grad().unwrap();
            let mut mask: Matrix<Owned<T>, DimDyn, D> = Matrix::alloc(input.shape());
            mask.to_ref_mut()
                .relu_backward_mask(&input.to_ref(), T::zero());
            let mask = Variable::from(mask);
            output_grad * mask
        };
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

#[must_use]
pub fn relu<T: Num, D: Device>(input: Variable<T, D>) -> Variable<T, D> {
    let output = alloc(input.get_shape());
    let relu = Relu::new(input, output.clone());
    relu.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(relu))));
    output
}

#[cfg(test)]
mod relu_test {

    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::Variable;

    use super::relu;

    fn relu_1d<D: Device>() {
        let x: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![-1., 0., 2., 3.], [2, 2]);
        let x_v = Variable::from(x);
        let y = relu(x_v.clone());
        y.backward();
        let ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![0., 0., 2., 3.], [2, 2]);
        let x_grad = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![0., 0., 1., 1.], [2, 2]);
        assert_val_eq!(y, ans, 1.0e-6);
        assert_val_eq_grad!(x_v, x_grad, 1.0e-6);
    }
    run_test!(relu_1d, relu_1d_cpu, relu_1d_nvidia);
}
