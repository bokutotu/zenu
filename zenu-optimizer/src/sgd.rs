use zenu_layer::Layer;
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
    fn update(&mut self, layers: &[Box<dyn Layer<T>>]) {
        let parameters = layers
            .into_iter()
            .flat_map(|layer| layer.parameters())
            .collect::<Vec<_>>();
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
    use zenu_layer::{layers::linear::Linear, Layer};

    use super::SGD;

    #[test]
    fn linear_1_layer() {
        // let mut layer = Linear::<f32>::new(1, 1);
        // let weight = from_vec(vec![1.], [1]);
        // let bias = from_vec(vec![1.], [1]);
        // layer.load_parameters(&[weight, bias]);
        // let sgd = SGD::new(0.1);
        // let input = from_vec(vec![1.], [1]);
        // let ans = from_vec(vec![4.], [1]);
        // let output = layer.call(input.clone());
    }
}
