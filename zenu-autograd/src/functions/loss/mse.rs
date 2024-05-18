use zenu_matrix::{device::Device, num::Num};

use crate::{functions::sum_to::sum_to, Variable};

pub fn mean_squared_error<T: Num, D: Device>(
    y_true: Variable<T, D>,
    y_pred: Variable<T, D>,
) -> Variable<T, D> {
    let batch_size = y_true.get_data().shape()[0];
    let diff = y_true - y_pred;
    let diff_squared = diff.clone() * diff;
    sum_to(diff_squared, []) / Variable::from(T::from_usize(batch_size))
}

#[cfg(test)]
mod mse {

    use zenu_matrix::{
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use super::mean_squared_error;

    fn batch_1<D: Device>() {
        let y_true = crate::creator::from_vec::from_vec(vec![1., 2., 3.], [1, 3]);
        let y_pred = crate::creator::from_vec::from_vec(vec![2., 3., 4.], [1, 3]);
        let mse = mean_squared_error(y_true.clone(), y_pred.clone());
        mse.backward();
        let mse_ans = Matrix::<Owned<f64>, DimDyn, D>::from_vec(vec![3.], []);
        assert_val_eq!(mse, mse_ans, 1e-6);

        let y_true_grad_ans =
            Matrix::<Owned<f64>, DimDyn, D>::from_vec(vec![-2., -2., -2.], [1, 3]);
        let y_pred_grad_ans = Matrix::<Owned<f64>, DimDyn, D>::from_vec(vec![2., 2., 2.], [1, 3]);
        assert_val_eq_grad!(y_true, y_true_grad_ans, 1e-6);
        assert_val_eq_grad!(y_pred, y_pred_grad_ans, 1e-6);
    }
    run_test!(batch_1, batch_1_cpu, batch_1_nvidia);
}
