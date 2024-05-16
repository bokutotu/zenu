use zenu_matrix::{
    device::Device,
    dim::DimDyn,
    matrix::{Matrix, Owned},
    num::Num,
};

use crate::Variable;

pub fn zeros<T: Num, I: Into<DimDyn>, D: Device>(dim: I) -> Variable<T, D> {
    let matrix = Matrix::<Owned<T>, DimDyn, D>::zeros(dim.into());
    Variable::new(matrix)
}

pub fn zeros_like<T: Num, D: Device>(a: &Variable<T, D>) -> Variable<T, D> {
    zeros(a.get_data().shape())
}
