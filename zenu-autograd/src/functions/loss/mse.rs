use zenu_matrix::{matrix::MatrixBase, num::Num};

use crate::{functions::sum_to::sum_to, Variable};

pub fn mean_squared_error<T: Num, D: Device>(y_true: Variable<T, D>, y_pred: Variable<T, D>) -> Variable<T, D> {
    let batch_size = y_true.get_data().shape()[0];
    let diff = y_true - y_pred;
    let diff_squared = diff.clone() * diff;
    sum_to(diff_squared, []) / Variable::from(T::from_usize(batch_size))
}

#[cfg(test)]
mod mse {
    use zenu_matrix::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};

    use super::mean_squared_error;

    #[test]
    fn batch_1() {
        let y_true = crate::creator::from_vec::from_vec(vec![1., 2., 3.], [1, 3]);
        let y_pred = crate::creator::from_vec::from_vec(vec![2., 3., 4.], [1, 3]);
        let mse = mean_squared_error(y_true.clone(), y_pred.clone());
        mse.backward();
        let mse_data = mse.get_data();
        let mse_ans = OwnedMatrixDyn::from_vec(vec![3.], []);
        let diff = mse_data - mse_ans;
        assert!(diff.asum() < 1e-6);
        let y_true_grad_ans = OwnedMatrixDyn::from_vec(vec![-2., -2., -2.], [1, 3]);
        let y_pred_grad_ans = OwnedMatrixDyn::from_vec(vec![2., 2., 2.], [1, 3]);
        let y_true_grad = y_true.get_grad().unwrap().get_data();
        let y_pred_grad = y_pred.get_grad().unwrap().get_data();
        let diff_y_true_grad = y_true_grad - y_true_grad_ans;
        let diff_y_pred_grad = y_pred_grad - y_pred_grad_ans;
        let diff_y_true_grad_asum = diff_y_true_grad.asum();
        let diff_y_pred_grad_asum = diff_y_pred_grad.asum();
        assert!(diff_y_true_grad_asum < 1e-6);
        assert!(diff_y_pred_grad_asum < 1e-6);
    }
}
