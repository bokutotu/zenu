use zenu_autograd::Variable;
use zenu_matrix::{
    matrix::ToViewMutMatrix, num::Num, operation::basic_operations::MatrixSubAssign,
};

use crate::Optimizer;

pub struct SGD<T: Num> {
    pub learning_rate: T,
}

impl<T: Num> SGD<T> {
    pub fn new(learning_rate: T) -> Self {
        Self { learning_rate }
    }
}

impl<T: Num> Optimizer<T> for SGD<T> {
    fn update(&self, parameters: &[Variable<T>]) {
        let parameters = parameters
            .into_iter()
            .filter(|parameter| parameter.get_grad().is_some())
            .collect::<Vec<_>>();
        parameters.into_iter().for_each(|parameter| {
            let grad = parameter.get_grad().unwrap().get_data();
            let mut data = parameter.get_data_mut();
            let update_data = grad * self.learning_rate;
            data.to_view_mut().sub_assign(update_data);
        });
    }
}

#[cfg(test)]
mod sgd {
    use zenu_autograd::creator::from_vec::from_vec;
    use zenu_matrix::{matrix::OwnedMatrix, matrix_impl::OwnedMatrixDyn, operation::asum::Asum};

    use crate::Optimizer;

    use super::SGD;

    #[test]
    fn linear_1_layer() {
        let variable = from_vec(vec![1., 2., 3., 4., 5., 6.], [3, 2]);
        variable.set_grad(from_vec(vec![1., 2., 3., 4., 5., 6.], [3, 2]));
        let sgd = SGD::new(1.);
        sgd.update(&[variable.clone()]);
        let data = variable.get_data();
        let ans = OwnedMatrixDyn::from_vec(vec![0., 0., 0., 0., 0., 0.], [3, 2]);
        let diff = data - ans;
        let diff_asum = diff.asum();
        assert_eq!(diff_asum, 0.0);
    }
}
