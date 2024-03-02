use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    dim::{DimDyn, DimTrait},
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory_impl::{ViewMem, ViewMutMem},
    num::Num,
    operation::{copy_from::CopyFrom, sum::MatrixSum, zeros::Zeros},
};

use crate::{Function, Variable, VariableWeak};

use super::broadcast::broadcast;

struct SumTo<T: Num> {
    x: Variable<T>,
    output: VariableWeak<T>,
}

impl<T: Num> SumTo<T> {
    pub fn new(x: Variable<T>, output: Variable<T>) -> Self {
        let output = output.downgrade();
        Self { x, output }
    }
}

fn inner<T: Num>(source: Matrix<ViewMem<T>, DimDyn>, target: Matrix<ViewMutMem<T>, DimDyn>) {
    if source.shape().len() < target.shape().len() {
        panic!("source.shape().len() < target.shape().len()");
    }

    let diff_len = source.shape().len() - target.shape().len();
    if diff_len == 0 {
        let mut target = target;
        target.to_view_mut().copy_from(&source.to_view());
        return;
    }

    if !source.shape().is_include(&target.shape()) {
        panic!("!source.shape().is_include(target.shape())");
    }

    if diff_len == 1 {
        let mut target = target;
        let ans = source.to_view().sum(0);
        target.to_view_mut().copy_from(&ans.to_view());
    } else {
        inner(source.to_view().sum(0).to_view(), target);
    }
}

impl<T: Num> Function<T> for SumTo<T> {
    fn forward(&self) {
        inner(
            self.x.get_data().to_view(),
            self.output.upgrade().unwrap().get_data_mut().to_view_mut(),
        );
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let output_grad = output.get_grad().clone().unwrap();
        let x_grad = broadcast(output_grad.clone(), self.x.get_data().shape());
        self.x.set_grad(x_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.x.clone()]
    }
}

pub fn sum_to<T: Num>(x: Variable<T>, shape: DimDyn) -> Variable<T> {
    let output = Variable::new(Zeros::zeros(shape));
    let sum_to = SumTo::new(x, output.clone());
    sum_to.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(sum_to))));
    output
}

#[cfg(test)]
mod sum_to {
    use zenu_matrix::{
        dim::DimDyn,
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::Matrix,
        memory_impl::OwnedMem,
        operation::{asum::Asum, ones::Ones},
    };

    use crate::Variable;

    use super::sum_to;

    #[test]
    fn sum_to_2d_1d() {
        let x: Matrix<OwnedMem<f32>, DimDyn> =
            Matrix::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [2, 3]);
        let x = Variable::from(x);
        let y = sum_to(x.clone(), DimDyn::new(&[3]));
        let forward_ans: Matrix<OwnedMem<f32>, DimDyn> = Matrix::from_vec(vec![2.0, 4.0, 6.0], [3]);
        let diff = y.get_data().to_view() - forward_ans.to_view();
        assert!(diff.asum() == 0.);

        y.backward();
        let x_grad: Matrix<OwnedMem<f32>, DimDyn> = Ones::ones([2, 3]);
        x.with_grad_data(|grad| {
            println!("grad: {:?}", grad.to_view());
            let diff = grad.to_view() - x_grad.to_view();
            assert!(diff.asum() == 0.);
        });
    }
}
