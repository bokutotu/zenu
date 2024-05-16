use std::{cell::RefCell, ops::Add, rc::Rc};

use zenu_matrix::{device::Device, num::Num};

use crate::{creator::zeros::zeros, Function, Variable, VariableWeak};

use super::{output_shape, sum_to::sum_to};

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
        let x_shape = self.x.get_data().shape();
        let y_shape = self.y.get_data().shape();
        let output = self.output.upgrade().unwrap();
        let grad = output.get_grad().clone().unwrap();
        self.x.set_grad(sum_to(grad.clone(), x_shape));
        self.y.set_grad(sum_to(grad, y_shape));
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone(), self.y.clone()]
    }
}

fn add<T: Num, D: Device>(x: Variable<T, D>, y: Variable<T, D>) -> Variable<T, D> {
    let output_shape = output_shape(&x, &y);
    let output = zeros(output_shape);
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

    use crate::Variable;

    fn add<D: Device>() {
        let x: Matrix<Owned<f32>, DimDyn, D> = Matrix::ones([100, 200]);
        let y: Matrix<Owned<f32>, DimDyn, D> = Matrix::ones([20, 100, 200]);
        let x_val = Variable::new(x);
        let y_val = Variable::new(y);
        let z = x_val.clone() + y_val.clone();
        z.backward();
        let z_data = z.get_data();
        let ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::ones([20, 100, 200]).to_ref() * 2.0;
        let diff = z_data.to_ref() - ans.to_ref();
        let diff_sum = diff.asum();
        assert!(diff_sum < 1e-6);

        x_val.with_grad_data(|grad| {
            let grad = grad.to_ref();
            let ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::ones([100, 200]).to_ref() * 20.;
            let diff = grad - ans.to_ref();
            let diff_sum = diff.asum();
            assert!(diff_sum < 1e-6);
        });
        y_val.with_grad_data(|grad| {
            let grad = grad.to_ref();
            let ans: Matrix<Owned<f32>, DimDyn, D> = Matrix::ones([20, 100, 200]);
            let diff = grad - ans.to_ref();
            let diff_sum = diff.asum();
            assert!(diff_sum < 1e-6);
        });
    }
    #[test]
    fn add_cpu() {
        add::<zenu_matrix::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn add_cuda() {
        add::<zenu_matrix::device::nvidia::Nvidia>();
    }
}
