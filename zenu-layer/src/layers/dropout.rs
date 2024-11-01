use std::{cell::RefCell, collections::HashMap, rc::Rc};

use zenu_autograd::{
    nn::dropout::{dropout, DropoutConfig},
    Variable,
};
use zenu_matrix::{
    device::Device,
    dim::{DimDyn, DimTrait},
    num::Num,
};

use crate::{Module, Parameters};

pub struct Dropout<T: Num, D: Device> {
    config: DropoutConfig<T, D>,
    input_shape: Option<Rc<RefCell<DimDyn>>>,
    raio: f32,
}

impl<T: Num, D: Device> Module<T, D> for Dropout<T, D> {
    type Input = Variable<T, D>;
    type Output = Variable<T, D>;
    fn call(&self, input: Variable<T, D>) -> Variable<T, D> {
        if self.input_shape.as_ref().unwrap().borrow().slice() != input.get_shape().slice() {
            todo!();
        }
        dropout(input, self.raio, Some(self.config.clone()))
    }
}

impl<T: Num, D: Device> Dropout<T, D> {
    #[must_use]
    pub fn new(rate: f32) -> Self {
        let config = DropoutConfig::new(rate);
        Self {
            config,
            input_shape: None,
            raio: rate,
        }
    }

    pub fn gpu_init(&self, shape: DimDyn) {
        self.config.gpu_init(shape);
    }
}

impl<T: Num, D: Device> Parameters<T, D> for Dropout<T, D> {
    fn weights(&self) -> HashMap<String, Variable<T, D>> {
        HashMap::new()
    }

    fn biases(&self) -> HashMap<String, Variable<T, D>> {
        HashMap::new()
    }
}
