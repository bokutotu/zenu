use std::{cell::RefCell, collections::HashMap, rc::Rc};

use zenu_autograd::{creator::zeros::zeros_like, Variable};
use zenu_layer::Parameters;
use zenu_matrix::{device::Device, num::Num};

use crate::Optimizer;

pub struct Adam<T: Num, D: Device> {
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    step: Rc<RefCell<usize>>,
    pub m: HashMap<String, Variable<T, D>>,
    pub v: HashMap<String, Variable<T, D>>,
}

impl<T: Num, D: Device, P: Parameters<T, D>> Optimizer<T, D, P> for Adam<T, D> {
    fn update(&self, parameters: &P) {
        *self.step.borrow_mut() += 1;
        let step = T::from_usize(*self.step.borrow());

        let beta1_t = self.beta1.powf(step);
        let beta2_t = self.beta2.powf(step);

        let parameters = parameters
            .parameters()
            .iter()
            .filter_map(|(key, value)| {
                value
                    .get_grad()
                    .map(|grad| (key.clone(), (value.clone(), grad.clone())))
            })
            .collect::<Vec<_>>();

        for (k, (data, grad)) in &parameters {
            let v = self.v.get(k).unwrap();
            let m = self.m.get(k).unwrap();
            let mut v = v.get_as_mut();
            let mut m = m.get_as_mut();
            let grad = grad.get_as_ref();

            m *= self.beta1;
            m += grad.to_ref() * (T::one() - self.beta1);

            v *= self.beta2;
            v += grad.to_ref() * grad.to_ref() * (T::one() - self.beta2);

            let m_hat = m.clone() / (T::one() - beta1_t);
            let v_hat = v.clone() / (T::one() - beta2_t);

            let m_v_hat = m_hat / (v_hat.sqrt() + self.epsilon);
            let lr_mv_hat = m_v_hat * self.learning_rate;

            data.get_as_mut().sub_assign(&lr_mv_hat.to_ref());
        }
    }
}

impl<T: Num, D: Device> Adam<T, D> {
    pub fn new(
        learning_rate: T,
        beta1: T,
        beta2: T,
        epsilon: T,
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
            step: Rc::new(RefCell::new(0)),
            m,
            v,
        }
    }
}
