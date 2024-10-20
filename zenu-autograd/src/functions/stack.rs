use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    num::Num,
    operation::{split::split as split_matrix, stack::stack as stack_matrices},
};

use crate::{
    creator::{alloc::alloc, zeros::zeros},
    Function, Variable, VariableWeak,
};

struct Stack<T: Num, D: Device> {
    vars: Vec<Variable<T, D>>,
    output: VariableWeak<T, D>,
    axis: usize,
}

impl<T: Num, D: Device> Function<T, D> for Stack<T, D> {
    fn forward(&self) {
        let matrices = self
            .vars
            .iter()
            .map(|v| v.get_data().clone())
            .collect::<Vec<_>>();
        let matrix = stack_matrices(&matrices, self.axis);
        self.output.upgrade().unwrap().swap_inner(matrix);
    }

    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().unwrap().clone();
        let input_grads = stack_grad(output_grad, self.axis, self.vars.len());
        self.vars
            .iter()
            .zip(input_grads)
            .for_each(|(v, g)| v.set_grad(g));
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        self.vars.clone()
    }
}

struct StackGrad<T: Num, D: Device> {
    input: Variable<T, D>,
    outputs: Vec<VariableWeak<T, D>>,
    axis: usize,
}

impl<T: Num, D: Device> Function<T, D> for StackGrad<T, D> {
    fn forward(&self) {
        let input_mat = self.input.get_data();
        let splits = split_matrix(&input_mat, self.axis, self.outputs.len());
        for (output_weak, split) in self.outputs.iter().zip(splits) {
            let output = output_weak.upgrade().unwrap();
            output.swap_inner(split);
        }
    }

    fn backward(&self) {
        let output_grads = self
            .outputs
            .iter()
            .map(|v| v.upgrade().unwrap().get_grad().unwrap().clone())
            .collect::<Vec<_>>();

        let input_grad = stack(&output_grads, self.axis);
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

#[must_use]
pub fn stack<T: Num, D: Device>(vars: &[Variable<T, D>], axis: usize) -> Variable<T, D> {
    let mut output_shape = vars[0].get_shape();
    output_shape[axis] *= vars.len();
    let output = alloc(output_shape);

    let stack = Stack {
        vars: vars.to_vec(),
        output: output.clone().downgrade(),
        axis,
    };

    stack.forward();

    output.set_creator(Rc::new(RefCell::new(Box::new(stack))));
    output
}

fn stack_grad<T: Num, D: Device>(
    input: Variable<T, D>,
    axis: usize,
    num_splits: usize,
) -> Vec<Variable<T, D>> {
    let mut output_shape = input.get_shape();
    output_shape[axis] /= num_splits;

    let mut outputs = Vec::with_capacity(num_splits);
    for _ in 0..num_splits {
        outputs.push(zeros(output_shape));
    }

    let stack_grad = StackGrad {
        input,
        outputs: outputs.iter().map(|v| v.clone().downgrade()).collect(),
        axis,
    };

    stack_grad.forward();

    let layer = Rc::new(RefCell::new(Box::new(stack_grad) as Box<dyn Function<T, D>>));
    outputs.iter().for_each(|v| v.set_creator(layer.clone()));
    outputs
}

#[cfg(test)]
mod stack_test {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::{creator::zeros::zeros, Variable};

    #[expect(clippy::many_single_char_names)]
    fn stack_fwd<D: Device>() {
        let x_mat = Matrix::<Owned<f32>, _, D>::ones([2, 3]);
        let y_mat = Matrix::<Owned<f32>, _, D>::zeros([2, 3]);
        let t_mat = Matrix::<Owned<f32>, _, D>::from_vec(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [2, 6],
        );

        let x = Variable::new(x_mat);
        let y = Variable::new(y_mat);
        let t = Variable::new(t_mat);

        let z = super::stack(&[x.clone(), y.clone()], 1);
        let w = z.clone() * t;
        w.backward();

        let z_expected = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
            [2, 6],
        );

        assert_val_eq!(z, z_expected, 1e-5);

        let x_grad_expected =
            Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2., 3., 7., 8., 9.], [2, 3]);
        let y_grad_expected =
            Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![4., 5., 6., 10., 11., 12.], [2, 3]);

        assert_val_eq_grad!(x, x_grad_expected, 1e-5);
        assert_val_eq_grad!(y, y_grad_expected, 1e-5);
    }
    run_test!(stack_fwd, stack_fwd_cpu, stack_fwd_gpu);

    #[expect(clippy::many_single_char_names)]
    fn stack_bwd<D: Device>() {
        let x: Variable<f32, D> = zeros([2, 2]);
        let y: Variable<f32, D> = zeros([2, 2]);
        let z: Variable<f32, D> = zeros([2, 2]);
        let t: Variable<f32, D> = zeros([2, 2]);
        let u = x.clone() + y.clone();
        let v = z.clone() + t.clone();
        let w = super::stack(&[u.clone(), v.clone()], 0);
        w.backward();

        assert!(x.get_grad().is_some());
        assert!(y.get_grad().is_some());
        assert!(z.get_grad().is_some());
        assert!(t.get_grad().is_some());
    }
    run_test!(stack_bwd, stack_bwd_cpu, stack_bwd_gpu);
}
