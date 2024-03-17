use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    constructor::zeros::Zeros,
    dim::DimDyn,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    num::Num,
    operation::broadcast::Broadcast as B,
};

use crate::{Function, Variable, VariableWeak};

use super::sum_to::sum_to;

struct Broadcast<T: Num> {
    x: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num> Broadcast<T> {
    pub fn new(x: Variable<T>, output: Variable<T>) -> Self {
        let output = output.downgrade();
        Self { x, output }
    }
}

impl<T: Num> Function<T> for Broadcast<T> {
    fn forward(&self) {
        let output = self.output.upgrade().unwrap();
        let mut output = output.get_data_mut();
        let x = self.x.get_data();
        let x = x.to_view();
        let mut output = output.to_view_mut();
        B::broadcast(&mut output, &x);
    }

    fn backward(&self) {
        let x_shape = self.x.get_data().shape();
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad();
        let output_grad = output_grad.clone();
        let x_grad = sum_to(output_grad.unwrap(), x_shape);
        self.x.set_grad(x_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.x.clone()]
    }
}

pub fn broadcast<T: Num>(x: Variable<T>, shape: DimDyn) -> Variable<T> {
    let output = Variable::new(Zeros::zeros(shape));
    let broadcast = Broadcast::new(x, output.clone());
    broadcast.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(broadcast))));
    output
}

#[cfg(test)]
mod broadcast {
    use zenu_matrix::{
        dim::DimDyn,
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::Matrix,
        memory_impl::OwnedMem,
        operation::asum::Asum,
    };

    use crate::Variable;

    use super::broadcast;

    #[test]
    fn broadcast_2d_1d() {
        let x: Matrix<OwnedMem<f32>, DimDyn> = Matrix::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let x = Variable::from(x);
        let y = broadcast(x.clone(), DimDyn::new(&[3, 3]));
        let forward_ans: Matrix<OwnedMem<f32>, DimDyn> =
            Matrix::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [3, 3]);
        let diff = y.get_data().to_view() - forward_ans.to_view();
        assert!(diff.asum() == 0.);

        y.backward();
        let backward_ans: Matrix<OwnedMem<f32>, DimDyn> =
            Matrix::from_vec(vec![3.0, 3.0, 3.0], [3]);
        x.with_grad_data(|grad| {
            let diff = grad.to_view() - backward_ans.to_view();
            assert!(diff.asum() < 1e-6);
        });
    }

    #[test]
    fn broadcast_4d_2d() {
        let x: Matrix<OwnedMem<f32>, DimDyn> = Matrix::from_vec(vec![1.0, 2.0], [1, 2]);
        let x = Variable::from(x);
        let y = broadcast(x.clone(), DimDyn::new(&[2, 3, 1, 2]));
        let forward_ans: Matrix<OwnedMem<f32>, DimDyn> = Matrix::from_vec(
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            [2, 3, 1, 2],
        );
        let diff = y.get_data().to_view() - forward_ans.to_view();
        assert!(diff.asum() == 0.);

        y.backward();
        let backward_ans: Matrix<OwnedMem<f32>, DimDyn> = Matrix::from_vec(vec![6.0, 6.0], [1, 2]);
        x.with_grad_data(|grad| {
            let diff = grad.to_view() - backward_ans.to_view();
            assert!(diff.asum() < 1e-6);
        });
    }
}
