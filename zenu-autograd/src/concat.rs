use zenu_matrix::{concat::concat as c, device::Device, num::Num};

use crate::Variable;

#[must_use]
pub fn concat<T: Num, D: Device>(vars: &[Variable<T, D>]) -> Variable<T, D> {
    let matrix = vars
        .iter()
        .map(|v| v.get_data().clone())
        .collect::<Vec<_>>();
    Variable::from(c(&matrix))
}
