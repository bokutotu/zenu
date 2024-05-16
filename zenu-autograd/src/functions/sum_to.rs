use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{device::Device, dim::DimDyn, num::Num, operation::sum::sum_to as sum_to_func};

use crate::{creator::zeros::zeros, Function, Variable, VariableWeak};

use super::broadcast::broadcast;

struct SumTo<T: Num, D: Device> {
    x: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> SumTo<T, D> {
    pub fn new(x: Variable<T, D>, output: Variable<T, D>) -> Self {
        let output = output.downgrade();
        Self { x, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for SumTo<T, D> {
    fn forward(&self) {
        sum_to_func(
            self.x.get_data().to_ref(),
            self.output.upgrade().unwrap().get_data_mut().to_ref_mut(),
        );
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().clone().unwrap();
        let x_grad = broadcast(output_grad.clone(), self.x.get_data().shape());
        self.x.set_grad(x_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone()]
    }
}

pub fn sum_to<T: Num, I: Into<DimDyn>, D: Device>(x: Variable<T, D>, shape: I) -> Variable<T, D> {
    let shape = shape.into();
    let output = zeros(shape);
    let sum_to = SumTo::new(x, output.clone());
    sum_to.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(sum_to))));
    output
}

#[cfg(test)]
mod sum_to {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    use crate::{creator::ones::ones, Variable};

    use super::sum_to;

    fn sum_to_2d_1d<D: Device>() {
        // let x: Matrix<Owned<f32>, DimDyn, D> =
        //     Matrix::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [2, 3]);
        // let x = Variable::from(x);
        // let y = sum_to(x.clone(), DimDyn::new(&[3]));
        // let forward_ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![2.0, 4.0, 6.0], [3]);
        // let diff = y.get_data() - forward_ans;
        // assert!(diff.asum() == 0.);
        //
        // y.backward();
        // let x_grad: Matrix<Owned<f32>, DimDyn, D> = ones([2, 3]);
        // x.with_grad_data(|grad| {
        //     let diff = grad - x_grad;
        //     assert!(diff.asum() == 0.);
        // });
    }
    #[test]
    fn sum_to_2d_1d_cpu() {
        sum_to_2d_1d::<zenu_matrix::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn sum_to_2d_1d_cuda() {
        sum_to_2d_1d::<zenu_matrix::device::nvidia::Nvidia>();
    }
}
