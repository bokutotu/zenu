use zenu_matrix::{concat::concat as c, num::Num};

use crate::Variable;

pub fn concat<T: Num>(vars: &[Variable<T>]) -> Variable<T> {
    let matrix = vars.iter().map(|v| v.get_data()).collect::<Vec<_>>();
    Variable::from(c(&matrix))
}
