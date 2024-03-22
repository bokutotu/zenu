pub mod dataset;
pub mod mnist;

use zenu_autograd::Variable;
use zenu_matrix::num::Num;
use zenu_optimizer::Optimizer;

pub trait Model<T: Num> {
    fn predict(&self, inputs: &[Variable<T>]) -> Variable<T>;
}

pub fn update_parameters<T: Num, O: Optimizer<T>>(loss: Variable<T>, optimizer: &O) {
    loss.backward();
    let parameters = loss.get_all_trainable_variables();
    optimizer.update(&parameters);
    loss.get_all_trainable_variables()
        .iter()
        .for_each(|x| x.clear_grad());
}
