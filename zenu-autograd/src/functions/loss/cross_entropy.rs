use zenu_matrix::{device::Device, num::Num};

use crate::{
    functions::{log::log, softmax::softmax, sum_to::sum_to},
    Variable,
};

pub fn cross_entropy<T: Num, D: Device>(
    pred: Variable<T, D>,
    ans: Variable<T, D>,
) -> Variable<T, D> {
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
        device::Device,
        dim::DimDyn,
        matrix::{Matrix, Owned},
    };
    use zenu_test::{assert_val_eq, assert_val_eq_grad, run_test};

    use crate::{creator::from_vec::from_vec, Variable};

    use super::cross_entropy;

    fn cross_entropy_batch_size_1<D: Device>() {
        let pred = from_vec(vec![0.1, 0.9, 0.1, 0.1], [1, 4]);
        let ans = from_vec(vec![0.0, 1.0, 0.0, 0.0], [1, 4]);
        let loss = super::cross_entropy(pred.clone(), ans);
        loss.backward();
        let ans = Matrix::<Owned<f64>, DimDyn, D>::from_vec(vec![0.8536], &[]);
        assert_val_eq!(loss, ans, 1e-4);
        let pred_ans = Matrix::<Owned<f64>, DimDyn, D>::from_vec(
            vec![0.1914, -0.5741, 0.1914, 0.1914],
            [1, 4],
        );
        assert_val_eq_grad!(pred, pred_ans, 1e-4);
    }
    run_test!(
        cross_entropy_batch_size_1,
        cross_entropy_batch_size_1_cpu,
        cross_entropy_batch_size_1_nvidia
    );

    fn cross_entropy_batch_size_2<D: Device>() {
        let pred: Variable<f32, D> =
            from_vec(vec![0.1, 0.9, 0.1, 0.1, 0.01, 0.9, 0.2, 0.05], [2, 4]);
        let ans = from_vec(vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [2, 4]);
        let loss = cross_entropy(pred.clone(), ans.clone());
        loss.backward();
        let ans_ = Matrix::<Owned<f32>, DimDyn, D>::from_vec(vec![1.2957], []);
        assert_val_eq!(loss, ans_, 2e-5);
        let pred_grad_ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![
                0.0957, -0.2871, 0.0957, 0.0957, -0.4121, 0.2142, 0.1064, 0.0915,
            ],
            [2, 4],
        );
        assert_val_eq_grad!(pred, pred_grad_ans, 3e-4);
        let ans_grad_ans = Matrix::<Owned<f32>, DimDyn, D>::from_vec(
            vec![
                0.8268, 0.4268, 0.8268, 0.8268, 0.8689, 0.4239, 0.7739, 0.8489,
            ],
            [2, 4],
        );
        assert_val_eq_grad!(ans, ans_grad_ans, 3e-4);
    }
    run_test!(
        cross_entropy_batch_size_2,
        cross_entropy_batch_size_2_cpu,
        cross_entropy_batch_size_2_nvidia
    );
}
