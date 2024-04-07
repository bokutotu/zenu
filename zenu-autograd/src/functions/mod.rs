use zenu_matrix::{
    dim::{larger_shape, DimDyn},
    matrix::MatrixBase,
    num::Num,
};

use crate::Variable;

mod add;
mod mul;

pub mod activation;
pub mod batch_norm;
pub mod broadcast;
pub mod clip;
pub mod conv2d;
pub mod cosh;
pub mod div;
pub mod exp;
pub mod flatten;
pub mod log;
pub mod loss;
pub mod matmul;
pub mod powf;
pub mod reshape;
pub mod sinh;
pub mod softmax;
pub mod sub;
pub mod sum;
pub mod sum_to;
pub mod tanh;
pub mod transpose;

pub(crate) fn output_shape<T: Num>(x: &Variable<T>, y: &Variable<T>) -> DimDyn {
    let x_shape = x.get_data().shape();
    let y_shape = y.get_data().shape();
    larger_shape(x_shape, y_shape)
}
