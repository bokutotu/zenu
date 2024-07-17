use std::{cell::RefCell, ops::Add, rc::Rc};

use zenu_matrix::{device::Device, num::Num};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

use super::output_shape;

struct Addition<T: Num, D: Device> {
    x: Variable<T, D>,
    y: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Addition<T, D> {
    pub fn new(x: Variable<T, D>, y: Variable<T, D>, output: Variable<T, D>) -> Self {
        let output = output.downgrade();
        Self { x, y, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for Addition<T, D> {
    fn forward(&self) {
        let x = self.x.get_data();
        let y = self.y.get_data();
        let output = self.output.upgrade().unwrap();
        let output_mat = x.to_ref() + y.to_ref();
        output.get_data_mut().to_ref_mut().copy_from(&output_mat);
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let grad = output.get_grad().clone().unwrap();
        self.x.set_grad(grad.clone());
        self.y.set_grad(grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

fn add<T: Num, D: Device>(x: Variable<T, D>, y: Variable<T, D>) -> Variable<T, D> {
    let output_shape = output_shape(&x, &y);
    let output = alloc(output_shape);
    let add = Addition::new(x, y, output.clone());
    add.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(add))));
    output
}

impl<T: Num, D: Device> Add<Variable<T, D>> for Variable<T, D> {
    type Output = Variable<T, D>;

    fn add(self, other: Variable<T, D>) -> Self::Output {
        add(self, other)
    }
}

#[cfg(test)]
mod add {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };

    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::Variable;

    fn add<D: Device>() {
        let x: Matrix<Owned<f32>, DimDyn, D> = Matrix::ones([5, 10]);
        let y: Matrix<Owned<f32>, DimDyn, D> = Matrix::ones([2, 5, 10]);
        let x_val = Variable::new(x);
        let y_val = Variable::new(y);
        let z = x_val.clone() + y_val.clone();
        z.backward();
        let ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::ones([2, 5, 10]).to_ref() * 2.0;
        assert_val_eq!(z, ans, 1e-6);

        let x_grad_ans = Matrix::<_, DimDyn, _>::ones([5, 10]).to_ref() * 2.;
        let y_grad_ans = Matrix::<_, DimDyn, _>::ones([2, 5, 10]);
        assert_val_eq_grad!(x_val, x_grad_ans, 1e-6);
        assert_val_eq_grad!(y_val, y_grad_ans, 1e-6);
    }
    run_test!(add, add_cpu, add_gpu);

    fn add_2<D: Device>() {
        let x: Matrix<Owned<f32>, DimDyn, D> = Matrix::ones([100, 200]);
        let y: Matrix<Owned<f32>, DimDyn, D> = Matrix::ones([20, 100, 200]);
        let x_val = Variable::new(x);
        let y_val = Variable::new(y);
        let z = x_val.clone() + y_val.clone();
        z.backward();
        let ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::ones([20, 100, 200]).to_ref() * 2.0;
        assert_val_eq!(z, ans, 1e-6);

        let x_grad_ans = Matrix::<_, DimDyn, _>::ones([100, 200]).to_ref() * 20.;
        let y_grad_ans = Matrix::<_, DimDyn, _>::ones([20, 100, 200]);
        assert_val_eq_grad!(x_val, x_grad_ans, 1e-6);
        assert_val_eq_grad!(y_val, y_grad_ans, 1e-6);
    }
    run_test!(add_2, add_cpu_2, add_gpu_2);
}
