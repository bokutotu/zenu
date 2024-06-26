use rand::distributions::Distribution;
use rand_distr::StandardNormal;
use zenu_autograd::Variable;
use zenu_matrix::{device::Device, num::Num};

pub mod layers;

pub trait Layer<T: Num, D: Device> {
    fn init_parameters(&mut self, seed: Option<u64>)
    where
        StandardNormal: Distribution<T>;
    fn parameters(&self) -> Vec<Variable<T, D>>;
    fn load_parameters(&mut self, parameters: &[Variable<T, D>]);
    fn clear_gradients(&self) {
        for parameter in self.parameters() {
            parameter.clear_grad();
        }
    }
    fn call(&self, input: Variable<T, D>) -> Variable<T, D>;
    fn shape_check(&self, input: &Variable<T, D>);
}
