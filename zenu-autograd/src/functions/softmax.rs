use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    num::Num,
    operation::{softmax::SoftMax as S, zeros::Zeros},
};

use crate::{Function, Variable, VariableWeak};

struct SoftMax<T: Num> {
    input: Variable<T>,
    output: VariableWeak<T>,
    axis: usize,
}

impl<T: Num> SoftMax<T> {
    fn new(input: Variable<T>, output: VariableWeak<T>, axis: usize) -> Self {
        Self {
            input,
            output,
            axis,
        }
    }
}

impl<T: Num> Function<T> for SoftMax<T> {
    fn forward(&self) {
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        S::softmax_assign(
            &mut output.to_view_mut(),
            self.input.get_data().to_view(),
            self.axis,
        )
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().clone().unwrap();
        // let gy = y output * output_grad;
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.input.clone()]
    }
}

pub fn softmax<T: Num>(input: Variable<T>, axis: usize) -> Variable<T> {
    let output = Variable::new(Zeros::zeros(input.get_data().shape()));
    let softmax = SoftMax::new(input, output.clone().downgrade(), axis);
    softmax.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(softmax))));
    output
}
