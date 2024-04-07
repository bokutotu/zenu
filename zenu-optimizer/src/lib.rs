pub mod adam;
pub mod sgd;

use zenu_autograd::Variable;
use zenu_matrix::num::Num;

pub trait Optimizer<T: Num> {
    fn update(&self, parameters: &[Variable<T>]);
}
