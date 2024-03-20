use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{
    constructor::zeros::Zeros,
    dim::DimTrait,
    matrix::{MatrixBase, ToViewMatrix, ToViewMutMatrix},
    matrix_impl::OwnedMatrixDyn,
    num::Num,
    operation::{clip::Clip, log::Log, sum::sum_to},
};

use crate::{functions::clip::clip, Function, Variable, VariableWeak};

struct CrossEntropy<T: Num> {
    pred: Variable<T>,
    ans: Variable<T>,
    loss: VariableWeak<T>,
    epsilon: Option<T>,
}

impl<T: Num> CrossEntropy<T> {
    fn new(pred: Variable<T>, ans: Variable<T>, loss: Variable<T>, epsilon: Option<T>) -> Self {
        assert_eq!(pred.get_data().shape().len(), 2, "pred.shape().len() != 2");
        assert_eq!(ans.get_data().shape().len(), 2, "ans.shape().len() != 2");
        assert_eq!(
            pred.get_data().shape(),
            ans.get_data().shape(),
            "pred.shape() != ans.shape()"
        );
        assert_eq!(loss.get_data().shape().len(), 0, "loss.shape().len() != 0");
        let loss = loss.downgrade();
        CrossEntropy {
            pred,
            ans,
            loss,
            epsilon,
        }
    }
}

impl<T: Num> Function<T> for CrossEntropy<T> {
    fn forward(&self) {
        let pred = self.pred.get_data();
        let pred = pred.clip(
            self.epsilon.unwrap_or(T::epsilon()),
            T::one() - self.epsilon.unwrap_or(T::epsilon()),
        );
        let mut log = OwnedMatrixDyn::zeros_like(pred.to_view());
        log.log(pred.to_view());
        let x = self.ans.get_data() * log;
        let mut loss = self.loss.upgrade().unwrap().get_data();
        sum_to(x.to_view(), loss.to_view_mut())
    }

    fn backward(&self) {
        let pred = clip(
            self.pred.clone(),
            T::zero(),
            T::one() - self.epsilon.unwrap_or(T::epsilon()),
        );
        let grad = self.ans.clone() / pred.clone() * Variable::from(T::minus_one());
        let grad = grad / Variable::from(T::from_usize(pred.get_data().shape()[0]));
        self.pred.set_grad(grad);
    }

    fn get_inputs(&self) -> Vec<Variable<T>> {
        vec![self.pred.clone(), self.ans.clone()]
    }
}

pub fn cross_entropy<T: Num>(
    pred: Variable<T>,
    ans: Variable<T>,
    epsilon: Option<T>,
) -> Variable<T> {
    let loss = Variable::new(Zeros::zeros([]));
    let cross_entropy = CrossEntropy::new(pred, ans, loss.clone(), epsilon);
    cross_entropy.forward();
    loss.set_creator(Rc::new(RefCell::new(Box::new(cross_entropy))));
    loss
}

#[cfg(test)]
mod cross_entropy {
    use zenu_matrix::{
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::asum::Asum,
    };

    use crate::creator::from_vec::from_vec;

    #[test]
    fn cross_entropy_batch_size_1() {
        let pred = from_vec(vec![0.1, 0.9, 0.1, 0.1], [1, 4]);
        let ans = from_vec(vec![0.0, 1.0, 0.0, 0.0], [1, 4]);
        let loss = super::cross_entropy(pred.clone(), ans, None);
        loss.backward();
        let loss_data = loss.get_data();
        let ans = OwnedMatrixDyn::from_vec(vec![0.8536], &[]);
        let diff = loss_data.to_view() - ans.to_view();
        assert!(diff.asum() < 1e-4);
        let pred_grad = pred.get_grad().clone().unwrap();
        println!("{:?}", pred_grad);
        let pred_ans = OwnedMatrixDyn::from_vec(vec![0.1914, -0.5741, 0.1914, 0.1914], [1, 4]);
        let diff = pred_grad.get_data().to_view() - pred_ans.to_view();
        assert!(diff.asum() < 1e-4);
    }
}
