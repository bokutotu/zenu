use zenu_matrix::num::Num;

use crate::{functions::tanh::tanh, Variable};

pub fn sigmoid<T: Num, D: Device>(x: Variable<T, D>) -> Variable<T, D> {
    let one = T::one();
    let two = one + one;
    let half = one / two;
    let half = Variable::from(half);
    let x_half = x.clone() * half.clone();
    let x_half_tanh = tanh(x_half);
    half.clone() + half.clone() * x_half_tanh
}

#[cfg(test)]
mod sigmoid {
    use crate::creator::from_vec::from_vec;

    use super::*;
    use zenu_matrix::matrix::IndexItem;

    #[test]
    fn test_sigmoid() {
        let x = Variable::from(0.0);
        let y = sigmoid(x);
        assert_eq!(y.get_data().index_item([]) - 0.5 < 1e-6, true);
    }

    #[test]
    fn test_sigmoid_05() {
        let x = Variable::from(0.5);
        let y = sigmoid(x);
        assert_eq!(y.get_data().index_item([]) - 0.62245935 < 1e-6, true);
    }

    #[test]
    fn test_sigmoid_01() {
        let x = Variable::from(0.1);
        let y = sigmoid(x);
        assert_eq!(y.get_data().index_item([]) - 0.52497919 < 1e-6, true);
    }

    #[test]
    fn sigmoid_1d() {
        let x = from_vec(vec![0.0, 0.5, 0.1], [3]);
        let y = sigmoid(x);
        assert_eq!(y.get_data().index_item([0]) - 0.5 < 1e-6, true);
        assert_eq!(y.get_data().index_item([1]) - 0.62245935 < 1e-6, true);
        assert_eq!(y.get_data().index_item([2]) - 0.52497919 < 1e-6, true);
    }

    #[test]
    fn sigmoid_2d() {
        let x = from_vec(vec![0.0, 0.5, 0.1, 0.2, 0.3, 0.4], [2, 3]);
        let y = sigmoid(x);
        assert_eq!(y.get_data().index_item([0, 0]) - 0.5 < 1e-6, true);
        assert_eq!(y.get_data().index_item([0, 1]) - 0.62245935 < 1e-6, true);
        assert_eq!(y.get_data().index_item([0, 2]) - 0.52497919 < 1e-6, true);
        assert_eq!(y.get_data().index_item([1, 0]) - 0.54983399 < 1e-6, true);
        assert_eq!(y.get_data().index_item([1, 1]) - 0.57444252 < 1e-6, true);
        assert_eq!(y.get_data().index_item([1, 2]) - 0.59868766 < 1e-6, true);
    }
}
