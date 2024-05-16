use zenu_matrix::{
    device::Device,
    dim::DimDyn,
    matrix::{Matrix, Owned},
    num::Num,
};

use crate::Variable;

pub fn from_vec<T: Num, I: Into<DimDyn>, D: Device>(vec: Vec<T>, dim: I) -> Variable<T, D> {
    let matrix = Matrix::<Owned<T>, DimDyn, D>::from_vec(vec, dim.into());
    Variable::new(matrix)
}
