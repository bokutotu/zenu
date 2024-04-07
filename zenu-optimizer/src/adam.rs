use zenu_matrix::num::Num;

use crate::Optimizer;

pub struct Adam<T: Num> {
    pub learning_rate: T,
    pub alpha: T,
    pub beta1: T,
    pub beta2: T,
    pub epsilon: T,
    pub t: T,
}

impl<T: Num> Adam<T> {
    pub fn new(learning_rate: T, alpha: T, beta1: T, beta2: T, epsilon: T) -> Self {
        Self {
            learning_rate,
            alpha,
            beta1,
            beta2,
            epsilon,
            t: T::from_usize(1),
        }
    }
}

impl<T: Num> Optimizer<T> for Adam<T> {
    fn update(&self, parameters: &[zenu_autograd::Variable<T>]) {
        todo!();
    }
}
