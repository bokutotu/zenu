use zenu_autograd::Variable;
use zenu_matrix::num::Num;

pub trait Layer<T: Num> {
    fn forward(&self, input: Variable<T>) -> Variable<T>;
    fn init_parameters(&self);
    fn parameters(&self) -> Vec<Variable<T>>;
}
