use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    dim::DimDyn,
    index::{index_dyn_impl::Index, IndexAxisTrait},
    num::Num,
    shape_stride::ShapeStride,
};

use crate::{
    creator::{alloc::alloc, zeros::zeros},
    Function, Variable, VariableWeak,
};

struct IndexAxis<T: Num, D: Device> {
    input: Variable<T, D>,
    index: Index,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Function<T, D> for IndexAxis<T, D> {
    fn forward(&self) {
        let input_mat = self.input.get_data();
        let output_mat = input_mat.index_axis(self.index);
        self.output
            .upgrade()
            .unwrap()
            .get_as_mut()
            .copy_from(&output_mat);
    }

    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().unwrap();
        let input_grad = index_axis_grad(output_grad.clone(), self.index, self.input.get_shape());
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

struct IndexAxisGrad<T: Num, D: Device> {
    input: Variable<T, D>,
    index: Index,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Function<T, D> for IndexAxisGrad<T, D> {
    fn forward(&self) {
        let output = self.output.upgrade().unwrap();
        let output_mat = output.get_as_mut();
        let output_sliced = output_mat.index_axis_mut(self.index);
        let input_mat = self.input.get_as_ref();
        output_sliced.copy_from(&input_mat);
    }

    fn backward(&self) {
        let input_grad = zeros(self.input.get_shape());
        let output_grad = self.output.upgrade().unwrap().get_grad().unwrap();
        input_grad
            .get_as_mut()
            .index_axis_mut(self.index)
            .copy_from(&output_grad.get_as_ref());
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

#[must_use]
pub fn index_axis<T: Num, D: Device>(input: Variable<T, D>, index: Index) -> Variable<T, D> {
    let input_mat = input.get_as_ref();
    let shape = input_mat.shape();
    let stride = input_mat.stride();

    let output_shape: ShapeStride<DimDyn> = index.get_shape_stride(shape, stride);

    let output = alloc(output_shape.shape());

    let input_name = input.get_name();

    let index_axis = IndexAxis {
        input,
        index,
        output: output.clone().downgrade(),
    };

    index_axis.forward();

    output.set_creator(Rc::new(RefCell::new(Box::new(index_axis))));
    if let Some(name) = input_name {
        output.set_name(&format!("{name}_index_axis"));
    }
    output
}

fn index_axis_grad<T: Num, D: Device>(
    input: Variable<T, D>,
    index: Index,
    output_shape: DimDyn,
) -> Variable<T, D> {
    let output = zeros(output_shape);

    if let Some(name) = input.get_name() {
        output.set_name(&format!("{name}_index_axis_grad"));
    }

    let index_axis_grad = IndexAxisGrad {
        input,
        index,
        output: output.clone().downgrade(),
    };

    index_axis_grad.forward();

    output.set_creator(Rc::new(RefCell::new(Box::new(index_axis_grad))));
    output
}

#[cfg(test)]
mod index_axis_test {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        index::index_dyn_impl::Index,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::{creator::from_vec::from_vec, Variable};

    use super::index_axis;

    fn fwd<D: Device>() {
        let input: Variable<f32, D> = from_vec(vec![1., 2., 3., 4.], [2, 2]);
        let output = index_axis(input.clone(), Index::new(0, 0));
        output.backward();
        let expected_output = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 2.], [2]);
        let expected_input_grad =
            Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1., 1., 0., 0.], [2, 2]);
        assert_val_eq!(output, expected_output, 1e-5);
        assert_val_eq_grad!(input, expected_input_grad, 1e-5);
    }
    run_test!(fwd, fwd_cpu, fwd_gpu);

    fn two_layers_test<D: Device>() {
        let x: Variable<f32, D> = from_vec(vec![1., 2., 3., 4.], [2, 2]);
        let y: Variable<f32, D> = from_vec(vec![5., 6., 7., 8.], [2, 2]);
        let z = x.clone() * y.clone();
        let output = index_axis(z, Index::new(0, 0));
        output.backward();

        assert!(x.get_grad().is_some());
        assert!(y.get_grad().is_some());
    }
    run_test!(two_layers_test, two_layers_test_cpu, two_layers_test_gpu);
}
