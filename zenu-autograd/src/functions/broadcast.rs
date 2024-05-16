use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{device::Device, dim::DimDyn, num::Num};

use crate::{creator::zeros::zeros, Function, Variable, VariableWeak};

use super::sum_to::sum_to;

struct Broadcast<T: Num, D: Device> {
    x: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Broadcast<T, D> {
    pub fn new(x: Variable<T, D>, output: Variable<T, D>) -> Self {
        let output = output.downgrade();
        Self { x, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for Broadcast<T, D> {
    fn forward(&self) {
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        let x = self.x.get_data();
        let x = x.to_ref();
        let output = output.to_ref_mut();
        output.broadcast(&x);
    }

    fn backward(&self) {
        let x_shape = self.x.get_data().shape();
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().unwrap();
        let x_grad = sum_to(output_grad, x_shape);
        self.x.set_grad(x_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone()]
    }
}

pub fn broadcast<T: Num, D: Device>(x: Variable<T, D>, shape: DimDyn) -> Variable<T, D> {
    let output = zeros(shape);
    let broadcast = Broadcast::new(x, output.clone());
    broadcast.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(broadcast))));
    output
}

#[cfg(test)]
mod broadcast {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    use crate::Variable;

    use super::broadcast;

    fn broadcast_2d_1d<D: Device>() {
        let x: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let x = Variable::from(x);
        let y = broadcast(x.clone(), DimDyn::new(&[3, 3]));
        let forward_ans: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [3, 3]);
        let diff = y.get_data().to_ref() - forward_ans.to_ref();
        assert!(diff.asum() == 0.);

        y.backward();
        let backward_ans: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![3.0, 3.0, 3.0], [3]);
        x.with_grad_data(|grad| {
            let diff = grad.to_ref() - backward_ans.to_ref();
            assert!(diff.asum() < 1e-6);
        });
    }
    #[test]
    fn broadcast_2d_1d_cpu() {
        broadcast_2d_1d::<zenu_matrix::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn broadcast_2d_1d_cuda() {
        broadcast_2d_1d::<zenu_matrix::device::nvidia::Nvidia>();
    }

    fn broadcast_4d_2d<D: Device>() {
        let x: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1.0, 2.0], [1, 2]);
        let x = Variable::from(x);
        let y = broadcast(x.clone(), DimDyn::new(&[2, 3, 1, 2]));
        let forward_ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            [2, 3, 1, 2],
        );
        let diff = y.get_data().to_ref() - forward_ans.to_ref();
        assert!(diff.asum() == 0.);

        y.backward();
        let backward_ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![6.0, 6.0], [1, 2]);
        x.with_grad_data(|grad| {
            let diff = grad.to_ref() - backward_ans.to_ref();
            assert!(diff.asum() < 1e-6);
        });
    }
    #[test]
    fn broadcast_4d_2d_cpu() {
        broadcast_4d_2d::<zenu_matrix::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn broadcast_4d_2d_cuda() {
        broadcast_4d_2d::<zenu_matrix::device::nvidia::Nvidia>();
    }
}
