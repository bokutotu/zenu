use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{dim::DimDyn, num::Num, operation::sum::sum_to as sum_to_func};

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

impl<T: Num> Function<T> for SumTo<T> {
    fn forward(&self) {
        sum_to_func(
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

pub fn sum_to<T: Num, I: Into<DimDyn>>(x: Variable<T>, shape: I) -> Variable<T> {
    let shape = shape.into();
    let output = Variable::new(Zeros::zeros(shape));
    let sum_to = SumTo::new(x, output.clone());
    sum_to.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(sum_to))));
    output
}

#[cfg(test)]
mod sum_to {
    use zenu_matrix::{
        constructor::ones::Ones,
        dim::DimDyn,
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::Matrix,
        memory_impl::OwnedMem,
        operation::asum::Asum,
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
            let diff = grad.to_view() - x_grad.to_view();
            assert!(diff.asum() == 0.);
        });
    }
}
