use zenu_matrix::{
    device::Device,
    dim::{larger_shape, DimDyn},
    num::Num,
};

use crate::Variable;

mod add;
mod div;
mod mul;
mod sub;

pub mod broadcast;
pub mod clip;
pub mod cosh;
pub mod exp;
pub mod flatten;
pub mod index_axis;
pub mod log;
pub mod matmul;
pub mod powf;
pub mod reshape;
pub mod sinh;
pub mod slice;
pub mod stack;
pub mod sum;
pub mod sum_to;
pub mod tanh;
pub mod transpose;

pub(crate) fn output_shape<T: Num, D: Device>(x: &Variable<T, D>, y: &Variable<T, D>) -> DimDyn {
    let x_shape = x.get_data().shape();
    let y_shape = y.get_data().shape();
    larger_shape(x_shape, y_shape)
}
