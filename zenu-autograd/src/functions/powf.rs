use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{device::Device, num::Num};

use crate::{creator::zeros::zeros_like, Function, Variable, VariableWeak};

struct Powf<T: Num, D: Device> {
    input: Variable<T, D>,
    factor: T,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Function<T, D> for Powf<T, D> {
    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }

    fn forward(&self) {
        let x = self.input.get_data();
        let output = self.output.upgrade().unwrap();
        let mut y = output.get_data_mut();
        y.to_ref_mut().powf(&x, self.factor);
    }

    fn backward(&self) {
        let dx = powf(self.input.clone(), self.factor - T::one())
            * Variable::from(self.factor)
            * self.output.upgrade().unwrap().get_grad().unwrap();
        self.input.set_grad(dx);
    }
}

pub fn powf<T: Num, D: Device>(x: Variable<T, D>, factor: T) -> Variable<T, D> {
    let output = zeros_like(&x);
    let output_weak = output.clone().downgrade();
    let powf = Powf {
        input: x,
        factor,
        output: output_weak,
    };
    powf.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(powf))));
    output
}

#[cfg(test)]
mod powf {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::{creator::from_vec::from_vec, Variable};

    use super::powf;

    fn powf_<D: Device>() {
        let input: Variable<f32, D> = from_vec(vec![1.0, 2.0, 3.0], [3]);
        let output = powf(input.clone(), 2.0);
        output.backward();
        let expected = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1.0, 4.0, 9.0], [3]);
        assert_val_eq!(output, expected, 1e-6);
        let expected = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![2.0, 4.0, 6.0], [3]);
        assert_val_eq_grad!(input, expected, 1e-6);
    }
    run_test!(powf_, powf_cpu, pow_nvidia);
}
