use zenu_matrix::{dim::DimDyn, num::Num, operation::zeros::Zeros};

use crate::Variable;

pub fn zeros<T: Num, I: Into<DimDyn>>(dim: I) -> Variable<T> {
    let matrix = Zeros::zeros(dim.into());
    Variable::new(matrix)
}
