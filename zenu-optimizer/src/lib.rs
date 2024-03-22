pub mod sgd;

use zenu_layer::Layer;
use zenu_matrix::num::Num;

pub trait Optimizer<T: Num> {
    fn update(&mut self, parameters: &[Box<dyn Layer<T>>]);
}
