use zenu_matrix::{dim::DimDyn, matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, num::Num};

use crate::Variable;

pub fn from_vec<T: Num, I: Into<DimDyn>>(vec: Vec<T>, dim: I) -> Variable<T> {
    let matrix = OwnedMatrixDyn::from_vec(vec, dim.into());
    Variable::new(matrix)
}
