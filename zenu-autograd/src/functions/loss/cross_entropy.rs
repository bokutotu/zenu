use zenu_matrix::{matrix::MatrixBase, num::Num};

use crate::{
    functions::{log::log, softmax::softmax, sum_to::sum_to},
    Variable,
};

pub fn cross_entropy<T: Num>(pred: Variable<T>, ans: Variable<T>) -> Variable<T> {
    let pred = softmax(pred, 1);
    let log = log(pred.clone());
    let y_log_pred = ans.clone() * log;
    let sum = sum_to(y_log_pred, &[] as &[usize]);
    let n = T::from_usize(pred.get_data().shape()[0]);
    let n = Variable::from(n);
    sum / n * Variable::from(T::minus_one())
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
        let loss = super::cross_entropy(pred.clone(), ans);
        println!("{:?}", loss);
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
