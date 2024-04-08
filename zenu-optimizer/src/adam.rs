use std::{cell::RefCell, rc::Rc};

use zenu_autograd::Variable;
use zenu_matrix::{
    constructor::zeros::Zeros,
    matrix::{MatrixBase, ToOwnedMatrix, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::OwnedMatrixDyn,
    num::Num,
    operation::{
        basic_operations::{MatrixPowf, MatrixSubAssign},
        copy_from::CopyFrom,
    },
};

use crate::Optimizer;

pub struct Adam<T: Num> {
    pub alpha: T,
    pub beta1: T,
    pub beta2: T,
    pub epsilon: T,
    pub v_t_1: Rc<RefCell<Vec<OwnedMatrixDyn<T>>>>,
    pub s_t_1: Rc<RefCell<Vec<OwnedMatrixDyn<T>>>>,
}

impl<T: Num> Adam<T> {
    pub fn new(alpha: T, beta1: T, beta2: T, epsilon: T) -> Self {
        Self {
            alpha,
            beta1,
            beta2,
            epsilon,
            v_t_1: Rc::new(RefCell::new(Vec::new())),
            s_t_1: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn init_v_t_1(&self, parameters: &[Variable<T>]) {
        // zero vvectorで埋める
        let v_t_1 = &mut *self.v_t_1.borrow_mut();
        v_t_1.clear();
        for p in parameters {
            v_t_1.push(OwnedMatrixDyn::zeros(p.get_data().shape()));
        }
    }

    fn init_s_t_1(&self, parameters: &[Variable<T>]) {
        // zero vvectorで埋める
        let s_t_1 = &mut *self.s_t_1.borrow_mut();
        s_t_1.clear();
        for p in parameters {
            s_t_1.push(OwnedMatrixDyn::zeros(p.get_data().shape()));
        }
    }

    fn update_v_t_1(&self, parameters: &[Variable<T>]) {
        let v_t_1 = &mut *self.v_t_1.borrow_mut();
        for (v, p) in v_t_1.iter_mut().zip(parameters) {
            let v_clone = v.to_owned_matrix();
            let mut v_mut = v.to_view_mut();
            v_mut.copy_from(
                &(v_clone * self.beta1
                    + p.get_grad().unwrap().get_data() * (T::one() - self.beta1)),
            );
        }
    }

    fn update_s_t_1(&self, parameters: &[Variable<T>]) {
        let s_t_1 = &mut *self.s_t_1.borrow_mut();
        for (s, p) in s_t_1.iter_mut().zip(parameters) {
            let s_clone = s.to_owned_matrix();
            let mut s_mut = s.to_view_mut();
            let mut p_cloned = p.get_data().to_owned();
            p_cloned
                .to_view_mut()
                .powf(p.get_data().to_view(), T::from_usize(2));
            s_mut.copy_from(&(s_clone * self.beta2 + p_cloned * (T::one() - self.beta2)));
        }
    }

    fn update_parameters(&self, parameters: &[Variable<T>]) {
        let v_t_1 = self.v_t_1.borrow();
        let s_t_1 = self.s_t_1.borrow();
        for ((p, v), s) in parameters.iter().zip(v_t_1.iter()).zip(s_t_1.iter()) {
            let v_hat = v.clone() / (T::one() - self.beta1);
            let s_hat = s.clone() / (T::one() - self.beta2);

            let diff_p = v_hat * self.beta2 / (s_hat * self.epsilon);
            p.get_data_mut().to_view_mut().sub_assign(diff_p);
        }
    }
}

impl<T: Num> Optimizer<T> for Adam<T> {
    fn update(&self, parameters: &[Variable<T>]) {
        if self.v_t_1.borrow().is_empty() {
            self.init_v_t_1(parameters);
        }
        if self.s_t_1.borrow().is_empty() {
            self.init_s_t_1(parameters);
        }

        self.update_parameters(parameters);
        self.update_v_t_1(parameters);
        self.update_s_t_1(parameters);
    }
}
