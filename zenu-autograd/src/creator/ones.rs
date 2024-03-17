use zenu_matrix::{constructor::zeros::Zeros, dim::DimDyn, matrix::MatrixBase, num::Num};

use crate::Variable;

pub fn ones<T: Num, I: Into<DimDyn>>(dim: I) -> Variable<T> {
    let matrix = Zeros::zeros(dim.into());
    Variable::new(matrix)
}

pub fn ones_like<T: Num>(a: &Variable<T>) -> Variable<T> {
    ones(a.get_data().shape())
}
