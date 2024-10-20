use zenu_matrix::{
    device::Device,
    dim::DimDyn,
    matrix::{Matrix, Owned},
    num::Num,
};

use crate::Variable;

pub fn alloc<T: Num, I: Into<DimDyn>, D: Device>(shape: I) -> Variable<T, D> {
    let matrix = Matrix::<Owned<T>, DimDyn, D>::alloc(shape);
    Variable::new(matrix)
}

#[must_use]
#[expect(clippy::module_name_repetitions)]
pub fn alloc_like<T: Num, D: Device>(a: &Variable<T, D>) -> Variable<T, D> {
    alloc(a.get_data().shape())
}
