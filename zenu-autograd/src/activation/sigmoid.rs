use zenu_matrix::{device::Device, num::Num};

use crate::{functions::tanh::tanh, Variable};

#[expect(clippy::needless_pass_by_value)]
#[must_use]
pub fn sigmoid<T: Num, D: Device>(x: Variable<T, D>) -> Variable<T, D> {
    let one = T::one();
    let two = one + one;
    let half = one / two;
    let half = Variable::from(half);
    let x_half = x.clone() * half.clone();
    let x_half_tanh = tanh(x_half);
    half.clone() + half.clone() * x_half_tanh
}

#[expect(clippy::unreadable_literal, clippy::excessive_precision)]
#[cfg(test)]
mod sigmoid {
    use zenu_test::run_test;

    use crate::creator::from_vec::from_vec;

    use super::*;

    fn test_sigmoid<D: Device>() {
        let x = Variable::<f32, D>::from(0.0);
        let y = sigmoid(x);
        assert!(y.get_data().index_item([]) - 0.5 < 1e-6);
    }
    run_test!(test_sigmoid, test_sigmoid_cpu, test_sigmoid_nvidia);

    fn test_sigmoid_05<D: Device>() {
        let x = Variable::<f32, D>::from(0.5);
        let y = sigmoid(x);
        assert!(y.get_data().index_item([]) - 0.62245935 < 1e-6);
    }
    run_test!(test_sigmoid_05, test_sigmoid_05_cpu, test_sigmoid_05_nvidia);

    fn test_sigmoid_01<D: Device>() {
        let x = Variable::<f32, D>::from(0.1);
        let y = sigmoid(x);
        assert!(y.get_data().index_item([]) - 0.52497919 < 1e-6);
    }
    run_test!(test_sigmoid_01, test_sigmoid_01_cpu, test_sigmoid_01_nvidia);

    fn sigmoid_1d<D: Device>() {
        let x: Variable<f64, D> = from_vec(vec![0.0, 0.5, 0.1], [3]);
        let y = sigmoid(x);
        assert!(y.get_data().index_item([0]) - 0.5 < 1e-6);
        assert!(y.get_data().index_item([1]) - 0.62245935 < 1e-6);
        assert!(y.get_data().index_item([2]) - 0.52497919 < 1e-6);
    }
    run_test!(sigmoid_1d, sigmoid_1d_cpu, sigmoid_1d_nvidia);

    fn sigmoid_2d<D: Device>() {
        let x: Variable<f32, D> = from_vec(vec![0.0, 0.5, 0.1, 0.2, 0.3, 0.4], [2, 3]);
        let y = sigmoid(x);
        assert!(y.get_data().index_item([0, 0]) - 0.5 < 1e-6,);
        assert!(y.get_data().index_item([0, 1]) - 0.62245935 < 1e-6,);
        assert!(y.get_data().index_item([0, 2]) - 0.52497919 < 1e-6,);
        assert!(y.get_data().index_item([1, 0]) - 0.54983399 < 1e-6,);
        assert!(y.get_data().index_item([1, 1]) - 0.57444252 < 1e-6,);
        assert!(y.get_data().index_item([1, 2]) - 0.59868766 < 1e-6,);
    }
    run_test!(sigmoid_2d, sigmoid_2d_cpu, sigmoid_2d_nvidia);
}
