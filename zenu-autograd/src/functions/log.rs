use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{device::Device, num::Num};

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
            .to_ref_mut()
            .log_array(&input.to_ref());
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap().get_grad().unwrap();
        self.input.set_grad(output / self.input.clone());
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

pub fn log<T: Num, D: Device>(x: Variable<T, D>) -> Variable<T, D> {
    let output = zeros_like(&x);
    let log = Log::new(x, output.clone());
    log.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(log))));
    output
}

#[cfg(test)]
mod log {

    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::creator::from_vec::from_vec;

    use super::log;

    fn log_1d<D: Device>() {
        let x = from_vec(vec![1., 2., 3., 4.], [4]);
        let y = log(x.clone());
        y.backward();
        let forward_ans: Matrix<Owned<f64>, DimDyn, D> = Matrix::from_vec(
            vec![
                0.,
                0.6931471805599453,
                1.0986122886681098,
                1.3862943611198906,
            ],
            [4],
        );
        let x_grad: Matrix<Owned<f64>, DimDyn, D> =
            Matrix::from_vec(vec![1., 0.5, 1. / 3., 0.25], [4]);
        assert_val_eq!(y, forward_ans, 1e-7);
        assert_val_eq_grad!(x, x_grad, 1e-7);
    }
    run_test!(log_1d, log_1d_cpu, log_1d_gpu);
}
