use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    device::Device,
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned},
    num::Num,
};

use crate::{creator::zeros::zeros, Function, Variable, VariableWeak};

struct Transpose<T: Num, D: Device> {
    x: Variable<T, D>,
    output: VariableWeak<T, D>,
}

impl<T: Num, D: Device> Transpose<T, D> {
    pub fn new(x: Variable<T, D>, output: Variable<T, D>) -> Self {
        let output = output.downgrade();
        Self { x, output }
    }
}

impl<T: Num, D: Device> Function<T, D> for Transpose<T, D> {
    // FIXME: メモリを使いまわす
    fn forward(&self) {
        let x = self.x.get_data();
        let mut out: Matrix<Owned<T>, DimDyn, D> = Matrix::zeros(x.shape());
        out.to_ref_mut().copy_from(&x.to_ref());
        out.transpose();
        let output = self.output.upgrade().unwrap();
        output.get_data_mut().to_ref_mut().copy_from(&out.to_ref());
    }

    // FIXME: メモリを使いまわす
    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let grad = output.get_grad().clone().unwrap();
        self.x.set_grad(transpose(grad));
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone()]
    }
}

pub fn transpose<T: Num, D: Device>(x: Variable<T, D>) -> Variable<T, D> {
    if x.get_data().shape().len() < 2 {
        panic!("Not implemented yet");
    }
    let output_shape = x.get_data().shape_stride().transpose().shape();
    let output = zeros(output_shape);
    let transpose = Transpose::new(x, output.clone());
    transpose.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(transpose))));
    output
}

pub struct TransposeByIndex<T: Num, D: Device> {
    x: Variable<T, D>,
    output: VariableWeak<T, D>,
    index: Vec<usize>,
}

impl<T: Num, D: Device> TransposeByIndex<T, D> {
    pub fn new(x: Variable<T, D>, output: Variable<T, D>, index: Vec<usize>) -> Self {
        let output = output.downgrade();
        Self { x, output, index }
    }
}

impl<T: Num, D: Device> Function<T, D> for TransposeByIndex<T, D> {
    fn forward(&self) {
        let x = self.x.get_data();
        let mut out: Matrix<Owned<T>, DimDyn, D> = Matrix::zeros(x.shape());
        out.to_ref_mut().copy_from(&x.to_ref());
        let out = out.transpose_by_index_new_matrix(&self.index);
        let output = self.output.upgrade().unwrap();
        output.get_data_mut().to_ref_mut().copy_from(&out.to_ref());
    }

    fn backward(&self) {
        let output = self.output.upgrade().unwrap();
        let grad = output.get_grad().clone().unwrap();
        // inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        let inv_axis = (0..self.index.len())
            .map(|ax| ax % self.index.len())
            .collect::<Vec<_>>();
        self.x.set_grad(transpose_by_index(grad, inv_axis));
    }

    fn get_inputs(&self) -> Vec<Variable<T, D>> {
        vec![self.x.clone()]
    }
}

pub fn transpose_by_index<T: Num, D: Device>(
    x: Variable<T, D>,
    index: Vec<usize>,
) -> Variable<T, D> {
    let input_shape = x.get_data().shape();
    let mut output_shape = input_shape;
    for i in 0..index.len() {
        output_shape[i] = input_shape[index[i]];
    }
    let output = zeros(output_shape);
    let transpose = TransposeByIndex::new(x, output.clone(), index);
    transpose.forward();
    output.set_creator(Rc::new(RefCell::new(Box::new(transpose))));
    output
}

#[cfg(test)]
mod transpose {
    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, run_test};

    use crate::Variable;

    use super::transpose;

    fn test_transpose<D: Device>() {
        let x: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1., 2., 3., 4., 5., 6.], [2, 3]);
        let x = Variable::from(x);
        let y = transpose(x);
        let ans: Matrix<Owned<f32>, DimDyn, D> =
            Matrix::from_vec(vec![1., 4., 2., 5., 3., 6.], [3, 2]);
        assert_val_eq!(y, ans, 1e-6);
    }
    run_test!(test_transpose, transpose_cpu, transpose_nvidia);
}
