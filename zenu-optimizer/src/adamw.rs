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
            .filter_map(|(key, param)| {
                if param.get_grad().is_some() {
                    Some((key.clone(), param.clone()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for (k, parameter) in params {
            let v_t = self.v.get(&k).unwrap();
            let m_t = self.m.get(&k).unwrap();
            let grad = parameter.get_grad().unwrap();
            let mut grad = grad.get_data_mut();
            let param_data = parameter.get_data();

            if weight_keys.contains(&k) {
                grad.to_ref_mut()
                    .add_assign(&(param_data.to_ref() * self.weight_decay).to_ref());
            }

            let mut m = m_t.get_data_mut();
            let mut v = v_t.get_data_mut();

            m.to_ref_mut().mul_scalar_assign(self.beta1);
            m.to_ref_mut()
                .add_assign(&(grad.to_ref() * (T::one() - self.beta1)).to_ref());

            v.to_ref_mut().mul_scalar_assign(self.beta2);
            v.to_ref_mut()
                .add_assign(&(grad.to_ref().sqrt() * (T::one() - self.beta2)).to_ref());

            let m_hat = m.to_ref() / (T::one() - beta1_t);
            let v_hat = v.to_ref() / (T::one() - beta2_t);

            let mut param_data_mut = parameter.get_data_mut();
            param_data_mut
                .to_ref_mut()
                .sub_assign(&(m_hat / (v_hat.sqrt() + self.epsilon) * self.learning_rate).to_ref());
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
