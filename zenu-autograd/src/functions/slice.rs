use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{device::Device, dim::DimDyn, index::SliceTrait, num::Num, slice::Slice as S};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

struct Slice<T: Num, D: Device> {
    input: Variable<T, D>,
    s: S,
    output: VariableWeak<T, D>,
}

struct SliceBackward<T: Num, D: Device> {
    input: Variable<T, D>,
    s: S,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Function<T, D> for Slice<T, D> {
    fn forward(&self) {
        let input_data = self.input.get_as_ref();
        let input_slice = input_data.slice_dyn(self.s);
        let output = self.output.upgrade().unwrap();
        output.get_as_mut().copy_from(&input_slice);
    }

    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().unwrap();
        let input_grad = slice_bkwd(output_grad, self.s, self.input.get_shape());
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

impl<T: Num, D: Device> Function<T, D> for SliceBackward<T, D> {
    fn forward(&self) {
        let output = self.output.upgrade().unwrap();
        output
            .get_as_mut()
            .slice_mut(self.s)
            .copy_from(&self.input.get_as_ref());
    }

    fn backward(&self) {
        let output_grad = self.output.upgrade().unwrap().get_grad().unwrap();
        let input_grad = slice(output_grad, self.s);
        self.input.set_grad(input_grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.input.clone()]
    }
}

#[must_use]
pub fn slice<T: Num, D: Device>(input: Variable<T, D>, s: S) -> Variable<T, D> {
    let output_shape = s
        .sliced_shape_stride(input.get_as_ref().shape(), input.get_as_ref().stride())
        .shape();
    let output = alloc(output_shape);
    let slice = Slice {
        input,
        s,
        output: output.clone().downgrade(),
    };

    slice.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(slice))));

    output
}

#[expect(clippy::large_types_passed_by_value)]
fn slice_bkwd<T: Num, D: Device>(
    input: Variable<T, D>,
    s: S,
    output_shape: DimDyn,
) -> Variable<T, D> {
    let output = alloc(output_shape);
    let slice_bkwd = SliceBackward {
        input,
        s,
        output: output.clone().downgrade(),
    };

    slice_bkwd.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(slice_bkwd))));

    output
}

#[cfg(test)]
mod slice_tests {
    use zenu_matrix::{device::Device, dim::DimDyn, matrix::Matrix, slice_dynamic};
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::creator::from_vec::from_vec;

    use super::slice;

    fn slice_2d<D: Device>() {
        let input = from_vec::<f32, _, D>(vec![1., 1., 1., 2., 2., 2., 3., 3., 3.], [3, 3]);
        let output = slice(input.clone(), slice_dynamic!(1.., ..));
        output.backward();

        let expected = Matrix::<_, DimDyn, _>::from_vec(vec![2., 2., 2., 3., 3., 3.], [2, 3]);
        assert_val_eq!(output, expected, 1e-5);

        let expected_grad =
            Matrix::<_, DimDyn, _>::from_vec(vec![0., 0., 0., 1., 1., 1., 1., 1., 1.], [3, 3]);
        assert_val_eq_grad!(input, expected_grad, 1e-5);
    }
    run_test!(slice_2d, slice_2d_cpu, slice_2d_gpu);
}
