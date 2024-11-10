use zenu_layer::Parameters;
use zenu_matrix::{device::Device, num::Num};

use crate::Optimizer;

pub struct SGD<T: Num, D: Device> {
    pub learning_rate: T,
    _device: std::marker::PhantomData<D>,
}

impl<T: Num, D: Device> SGD<T, D> {
    pub fn new(learning_rate: T) -> Self {
        Self {
            learning_rate,
            _device: std::marker::PhantomData,
        }
    }
}

impl<T: Num, D: Device, P: Parameters<T, D>> Optimizer<T, D, P> for SGD<T, D> {
    fn update(&self, parameters: &P) {
        for data in parameters.parameters().values() {
            if let Some(grad) = data.get_grad() {
                let update_data = grad.get_data().to_ref() * self.learning_rate;
                let mut data = data.get_data_mut();
                let mut data = data.to_ref_mut();
                data -= update_data;
            }
        }
    }
}
