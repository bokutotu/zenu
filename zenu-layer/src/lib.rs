use zenu_autograd::{Function, Variable};
use zenu_matrix::num::Num;

pub mod layers;

pub trait Layer<T: Num>: Function<T> {
    fn init_parameters(&self);
    fn parameters(&self) -> Vec<Variable<T>>;
}
