use zenu_matrix::{constructor::zeros::Zeros, dim::DimDyn, num::Num};

use crate::Variable;

pub fn ones<T: Num, I: Into<DimDyn>>(dim: I) -> Variable<T> {
    let matrix = Zeros::zeros(dim.into());
    Variable::new(matrix)
}
