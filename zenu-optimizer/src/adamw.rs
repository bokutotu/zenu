// use std::{cell::RefCell, rc::Rc};
//
// use zenu_autograd::Variable;
// use zenu_matrix::{
//     constructor::zeros::Zeros,
//     matrix::{MatrixBase, ToOwnedMatrix, ToViewMatrix, ToViewMutMatrix},
//     matrix_impl::OwnedMatrixDyn,
//     num::Num,
//     operation::basic_operations::{MatrixAddAssign, MatrixSubAssign},
// };
//
// pub struct AdamW<T: Num> {
//     pub alpha: T,
//     weight_decay: T,
//     beta1: T,
//     beta2: T,
//     epsilon: T,
//     m: Rc<RefCell<Vec<OwnedMatrixDyn<T>>>>,
//     v: Rc<RefCell<Vec<OwnedMatrixDyn<T>>>>,
//     theta: Rc<RefCell<Vec<OwnedMatrixDyn<T>>>>,
// }
//
// impl<T: Num> AdamW<T> {
//     pub fn new(alpha: T, weight_decay: T, beta1: T, beta2: T, epsilon: T) -> Self {
//         Self {
//             alpha,
//             weight_decay,
//             beta1,
//             beta2,
//             epsilon,
//             m: Rc::new(RefCell::new(Vec::new())),
//             v: Rc::new(RefCell::new(Vec::new())),
//             theta: Rc::new(RefCell::new(Vec::new())),
//         }
//     }
//
//     fn init_m(&self, parameters: &[Variable<T>]) {
//         let m = &mut *self.m.borrow_mut();
//         m.clear();
//         for p in parameters {
//             m.push(OwnedMatrixDyn::zeros(p.get_data().shape()));
//         }
//     }
//
//     fn init_v(&self, parameters: &[Variable<T>]) {
//         let v = &mut *self.v.borrow_mut();
//         v.clear();
//         for p in parameters {
//             v.push(OwnedMatrixDyn::zeros(p.get_data().shape()));
//         }
//     }
//
//     fn init_theta(&self, parameters: &[Variable<T>]) {
//         let theta = &mut *self.theta.borrow_mut();
//         theta.clear();
//         for p in parameters {
//             theta.push(OwnedMatrixDyn::zeros(p.get_data().shape()));
//         }
//     }
//
//     fn update_m(&self, parameters: &[Variable<T>]) {
//         let m = &mut *self.m.borrow_mut();
//         for (m, p) in m.iter_mut().zip(parameters) {
//             let mut p_g = p.get_grad().unwrap().get_data();
//             let m_c = m.to_owned_matrix();
//             p_g.sub_assign(m_c);
//             m.to_view_mut().add_assign(p_g * (T::one() - self.beta1));
//         }
//     }
//
//     fn update_v(&self, parameters: &[Variable<T>]) {
//         let v = &mut *self.v.borrow_mut();
//         for (v, p) in v.iter_mut().zip(parameters) {
//             let p_g = p.get_grad().unwrap().get_data();
//             let v_c = v.to_owned_matrix();
//             let mut p_g = p_g.to_view() * p_g.to_view();
//             p_g.sub_assign(v_c);
//             v.to_view_mut().add_assign(p_g * (T::one() - self.beta2));
//         }
//     }
//
//     fn
// }
