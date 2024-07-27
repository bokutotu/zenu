use std::{cell::RefCell, ops::Sub, rc::Rc};

use zenu_matrix::{device::Device, num::Num};

use crate::{creator::alloc::alloc, Function, Variable};

use super::output_shape;

pub struct SubFunc<T: Num, D: Device> {
    x: Variable<T, D>,
    y: Variable<T, D>,
    output: Variable<T, D>,
}

impl<T: Num, D: Device> Function<T, D> for SubFunc<T, D> {
    fn forward(&self) {
        let x = self.x.get_data();
        let y = self.y.get_data();
        let output = x.to_ref() - y.to_ref();
        self.output
            .get_data_mut()
            .to_ref_mut()
            .copy_from(&output.to_ref());
    }

    fn backward(&self) {
        let output_grad = self.output.get_grad().clone().unwrap();
        let x_grad = output_grad.clone();
        let y_grad = output_grad.clone() * Variable::from(T::minus_one());
        self.x.set_grad(x_grad);
        self.y.set_grad(y_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

pub fn sub<T: Num, D: Device>(x: Variable<T, D>, y: Variable<T, D>) -> Variable<T, D> {
    let output_shape = output_shape(&x, &y);
    let output = alloc(output_shape);
    let sub = SubFunc {
        x,
        y,
        output: output.clone(),
    };
    sub.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(sub))));
    output
}

impl<T: Num, D: Device> Sub<Variable<T, D>> for Variable<T, D> {
    type Output = Variable<T, D>;

    fn sub(self, rhs: Variable<T, D>) -> Self::Output {
        sub(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use zenu_matrix::{
        device::cpu::Cpu,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad};

    use super::*;

    #[test]
    fn sub() {
        let x = Variable::<f32, Cpu>::new(Matrix::from_vec(vec![1., 2., 3.], [3]));
        let y = Variable::new(Matrix::from_vec(vec![1., 2., 3.], [3]));
        let z = x.clone() - y.clone();
        let ans = Matrix::<Owned<_>, DimDyn, _>::zeros([3]);
        let ones = Matrix::<_, DimDyn, _>::ones([3]);
        let minus_ones = Matrix::<_, DimDyn, _>::from_vec(vec![-1., -1., -1.], [3]);
        assert_val_eq!(z.clone(), ans, 1e-4);
        z.backward();
        assert_val_eq_grad!(x, ones, 1e-4);
        assert_val_eq_grad!(y, minus_ones, 1e-4);
    }
}
