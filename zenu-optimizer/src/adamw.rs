use std::{cell::RefCell, collections::HashMap, rc::Rc};

use zenu_autograd::{creator::zeros::zeros_like, Variable};
use zenu_layer::Parameters;
use zenu_matrix::{device::Device, num::Num};

use crate::Optimizer;

pub struct AdamW<T: Num, D: Device> {
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    weight_decay: T,
    step: Rc<RefCell<T>>,
    m: HashMap<String, Variable<T, D>>,
    v: HashMap<String, Variable<T, D>>,
}

impl<T: Num, D: Device, P: Parameters<T, D>> Optimizer<T, D, P> for AdamW<T, D> {
    fn update(&self, parameters: &P) {
        let step = *self.step.borrow() + T::one();
        *self.step.borrow_mut() = step;

        let beta1_t = self.beta1.powf(step);
        let beta2_t = self.beta2.powf(step);

        let weight_keys: Vec<_> = parameters.weights().keys().cloned().collect();

        let params = parameters
            .parameters()
            .iter()
            .filter_map(|(key, value)| {
                value
                    .get_grad()
                    .map(|grad| (key.clone(), (value.clone(), grad.clone())))
            })
            .collect::<Vec<_>>();

        for (k, (data, grad)) in params {
            let m = self.m.get(&k).unwrap();
            let v = self.v.get(&k).unwrap();
            let mut m = m.get_as_mut();
            let mut v = v.get_as_mut();
            let grad = grad.get_as_ref();

            // Update m and v
            m *= self.beta1;
            m += grad.to_ref() * (T::one() - self.beta1);

            v *= self.beta2;
            v += grad.to_ref() * grad.to_ref() * (T::one() - self.beta2);

            let m_hat = m.clone() / (T::one() - beta1_t);
            let v_hat = v.clone() / (T::one() - beta2_t);

            let denom = v_hat.sqrt() + self.epsilon;
            let step_size = self.learning_rate;
            let update = m_hat / denom;

            if weight_keys.contains(&k) {
                data.get_as_mut().sub_assign(
                    &(data.get_as_ref() * self.learning_rate * self.weight_decay).to_ref(),
                );
            }

            data.get_as_mut().sub_assign(&(update * step_size).to_ref());
        }
    }
}
impl<T: Num, D: Device> AdamW<T, D> {
    pub fn new(
        learning_rate: T,
        beta1: T,
        beta2: T,
        epsilon: T,
        weight_decay: T,
        model: &impl Parameters<T, D>,
    ) -> Self {
        let m = model
            .parameters()
            .iter()
            .map(|(key, value)| (key.clone(), zeros_like(value)))
            .collect();
        let v = model
            .parameters()
            .iter()
            .map(|(key, value)| (key.clone(), zeros_like(value)))
            .collect();
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step: Rc::new(RefCell::new(T::zero())),
            m,
            v,
        }
    }
}
