use zenu_matrix::{dim::DimDyn, matrix::MatrixBase, num::Num};

use crate::Variable;

mod add;
mod mul;

pub mod broadcast;
pub mod matmul;
pub mod relu;
pub mod softmax;
pub mod sum_to;
pub mod transpose;

pub(crate) fn output_shape<T: Num>(x: &Variable<T>, y: &Variable<T>) -> DimDyn {
    if x.get_data().shape().is_include(&y.get_data().shape()) {
        x.get_data().shape()
    } else {
        y.get_data().shape()
    }
}
