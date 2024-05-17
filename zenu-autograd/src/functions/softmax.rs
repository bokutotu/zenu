use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    constructor::zeros::Zeros,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    num::Num,
    operation::softmax::SoftMax as S,
};

use crate::{Function, Variable, VariableWeak};

use super::sum::sum;

struct SoftMax<T: Num, D: Device> {
    input: Variable<T, D>,
    output: VariableWeak<T, D>,
    axis: usize,
}

impl<T: Num, D: Device> SoftMax<T, D> {
    fn new(input: Variable<T, D>, output: VariableWeak<T, D>, axis: usize) -> Self {
        Self {
            input,
            output,
            axis,
        }
    }
}

impl<T: Num, D: Device> Function<T, D> for SoftMax<T, D> {
    fn forward(&self) {
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        S::softmax_assign(
            &mut output.to_ref_mut(),
            self.input.get_data().to_ref(),
            self.axis,
        )
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let input_grad = output.clone() * output.clone().get_grad().unwrap();
        let sum_input_grad = sum(input_grad.clone(), self.axis, true);
        let input_grad = input_grad - (output * sum_input_grad);
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

pub fn softmax<T: Num, D: Device>(input: Variable<T, D>, axis: usize) -> Variable<T, D> {
    let output = Variable::new(Zeros::zeros(input.get_data().shape()));
    let softmax = SoftMax::new(input, output.clone().downgrade(), axis);
    softmax.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(softmax))));
    output
}

#[cfg(test)]
mod softmax {
    use zenu_matrix::{
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::asum::Asum,
    };

    use crate::Variable;

    use super::softmax;

    #[test]
    fn softmax_2d_1d() {
        let input = OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0, 4., 5., 6., 7., 8.], [2, 4]);
        let input = Variable::from(input);
        let output = softmax(input.clone(), 1);
        output.backward();
        let expected = OwnedMatrixDyn::from_vec(
            vec![
                0.0320586, 0.08714432, 0.23688284, 0.64391428, 0.0320586, 0.08714432, 0.23688284,
                0.64391428,
            ],
            [2, 4],
        );
        let diff = output.get_data().to_ref() - expected.to_ref();
        assert!(diff.asum() < 1e-6);
        let output =
            output * Variable::from(OwnedMatrixDyn::from_vec(vec![1.0, 2.0, 3.0, 4.0], [4]));
        output.backward();
        let grad = input.get_grad().unwrap();
        // [[-0.07991097 -0.13007621 -0.11670097  0.32668814]
        //[-0.07991097 -0.13007621 -0.11670097  0.32668814]]

        let ans = OwnedMatrixDyn::from_vec(
            vec![
                -0.07991097,
                -0.13007621,
                -0.11670097,
                0.32668814,
                -0.07991097,
                -0.13007621,
                -0.11670097,
                0.32668814,
            ],
            [2, 4],
        );
        let diff = grad.get_data().to_ref() - ans.to_ref();
        assert!(diff.asum() < 1e-6);
    }
}
