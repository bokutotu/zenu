use zenu_matrix::{matrix::MatrixBase, num::Num};

use crate::{
    functions::{log::log, softmax::softmax, sum_to::sum_to},
    Variable,
};

pub fn cross_entropy<T: Num, D: Device>(pred: Variable<T, D>, ans: Variable<T, D>) -> Variable<T, D> {
    let pred = softmax(pred, 1);
    let log = log(pred.clone());
    let y_log_pred = ans.clone() * log;
    let sum = sum_to(y_log_pred, &[] as &[usize]);
    let n = T::from_usize(pred.get_data().shape()[0]);
    let n = Variable::from(-n);
    sum / n
}

#[cfg(test)]
mod cross_entropy {
    use zenu_matrix::{
        matrix::{OwnedMatrix, ToViewMatrix},
        matrix_impl::OwnedMatrixDyn,
        operation::asum::Asum,
    };

    use crate::creator::from_vec::from_vec;

    use super::cross_entropy;

    #[test]
    fn cross_entropy_batch_size_1() {
        let pred = from_vec(vec![0.1, 0.9, 0.1, 0.1], [1, 4]);
        let ans = from_vec(vec![0.0, 1.0, 0.0, 0.0], [1, 4]);
        let loss = super::cross_entropy(pred.clone(), ans);
        loss.backward();
        let loss_data = loss.get_data();
        let ans = OwnedMatrixDyn::from_vec(vec![0.8536], &[]);
        let diff = loss_data.to_ref() - ans.to_ref();
        assert!(diff.asum() < 1e-4);
        let pred_grad = pred.get_grad().clone().unwrap();
        let pred_ans = OwnedMatrixDyn::from_vec(vec![0.1914, -0.5741, 0.1914, 0.1914], [1, 4]);
        let diff = pred_grad.get_data().to_ref() - pred_ans.to_ref();
        assert!(diff.asum() < 1e-4);
    }

    #[test]
    fn cross_entropy_batch_size_2() {
        let pred = from_vec(vec![0.1, 0.9, 0.1, 0.1, 0.01, 0.9, 0.2, 0.05], [2, 4]);
        let ans = from_vec(vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [2, 4]);
        let loss = cross_entropy(pred.clone(), ans.clone());
        loss.backward();
        let loss_data = loss.get_data();
        let ans_ = OwnedMatrixDyn::from_vec(vec![1.2975], []);
        let diff = loss_data.to_ref() - ans_.to_ref();
        assert!(diff.asum() < 1e-4);
        let pred_grad_ans = OwnedMatrixDyn::from_vec(
            vec![
                0.0957, -0.2871, 0.0957, 0.0957, -0.4121, 0.2142, 0.1064, 0.0915,
            ],
            [2, 4],
        );
        let pred_grad = pred.get_grad().clone().unwrap().get_data();
        let diff = pred_grad.to_ref() - pred_grad_ans.to_ref();
        assert!(diff.asum() < 5e-4);
        let ans_grad_ans = OwnedMatrixDyn::from_vec(
            vec![
                0.8268, 0.4268, 0.8268, 0.8268, 0.8689, 0.4239, 0.7739, 0.8489,
            ],
            [2, 4],
        );
        let ans_grad_pred = ans.get_grad().unwrap().get_data();
        let diff = ans_grad_pred.to_ref() - ans_grad_ans.to_ref();
        assert!(diff.asum() < 5e-4);
    }
}
