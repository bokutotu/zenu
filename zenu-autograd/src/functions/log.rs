use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device, num::Num
};

use crate::{creator::zeros::zeros_like, Function, Variable, VariableWeak};

struct Log<T: Num, D: Device> {
    input: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Log<T, D> {
    pub fn new(input: Variable<T, D>, output: Variable<T, D>) -> Self {
        assert_eq!(
            input.get_data().shape(),
            output.get_data().shape(),
            "input and output shape must be same"
        );
        let output = output.downgrade();
        Self { input, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for Log<T, D> {
    fn forward(&self) {
        let input = self.input.get_data();
        self.output
            .upgrade()
            .unwrap()
            .get_data_mut()
            .to_view_mut()
            .log(input.to_view());
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap().get_grad().unwrap();
        self.input.set_grad(output / self.input.clone());
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

pub fn log<T: Num>(x: Variable<T>) -> Variable<T> {
    let output = zeros_like(&x);
    let log = Log::new(x, output.clone());
    log.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(log))));
    output
}

#[cfg(test)]
mod log {
    use zenu_matrix::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};

    use crate::creator::from_vec::from_vec;

    use super::log;

    #[test]
    fn log_1d() {
        let x = from_vec(vec![1., 2., 3., 4.], [4]);
        let y = log(x.clone());
        y.backward();
        let forward_ans = OwnedMatrixDyn::from_vec(
            vec![
                0.,
                0.6931471805599453,
                1.0986122886681098,
                1.3862943611198906,
            ],
            [4],
        );
        let forward_result = y.get_data();
        let diff = forward_ans - forward_result;
        assert!(diff.asum() < 1e-7);
        let grad = x.get_grad().unwrap().get_data();
        let grad_ans = OwnedMatrixDyn::from_vec(vec![1., 0.5, 1. / 3., 0.25], [4]);
        let diff = grad_ans - grad;
        assert!(diff.asum() < 1e-7);
    }
}
