use zenu_autograd::Variable;
use zenu_layer::Layer;
use zenu_matrix::num::Num;

pub trait Model<T: Num> {
    fn forward(&self, inputs: &[Variable<T>]) -> Variable<T>;
    fn parameters(&self) -> Vec<Box<dyn Layer<T>>>;
}

pub trait Optimizer<T: Num> {
    fn update(&mut self, parameters: &[Box<dyn Layer<T>>]);
}
