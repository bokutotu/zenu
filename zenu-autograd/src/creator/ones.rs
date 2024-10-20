use zenu_matrix::{
    device::Device,
    dim::DimDyn,
    matrix::{Matrix, Owned},
    num::Num,
};

use crate::Variable;

pub fn ones<T: Num, I: Into<DimDyn>, D: Device>(dim: I) -> Variable<T, D> {
    let matrix = Matrix::<Owned<T>, DimDyn, D>::ones(dim.into());
    Variable::new(matrix)
}

#[expect(clippy::module_name_repetitions)]
#[must_use]
pub fn ones_like<T: Num, D: Device>(a: &Variable<T, D>) -> Variable<T, D> {
    ones(a.get_data().shape())
}
