use crate::Module;
use rand_distr::{Distribution, StandardNormal};
use zenu_autograd::{
    creator::{rand::normal, zeros::zeros},
    functions::matmul::matmul,
    Variable,
};
use zenu_matrix::{device::Device, num::Num};

pub struct Linear<T: Num, D: Device> {
    in_features: usize,
    out_features: usize,
    weight: Variable<T, D>,
    bias: Option<Variable<T, D>>,
}

impl<T: Num, D: Device> Module<T, D> for Linear<T, D> {
    fn call(&self, input: Variable<T, D>) -> Variable<T, D> {
        let output = matmul(input, self.weight.clone());
        if let Some(bias) = &self.bias {
            output + bias.clone()
        } else {
            output
        }
    }
}

impl<T: Num, D: Device> Linear<T, D> {
    #[must_use]
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self
    where
        StandardNormal: Distribution<T>,
    {
        let weight = normal(T::zero(), T::one(), None, [in_features, out_features]);
        let bias = if use_bias {
            Some(zeros([out_features]))
        } else {
            None
        };
        Self {
            in_features,
            out_features,
            weight,
            bias,
        }
    }
}

#[cfg(test)]
mod linear {
    use zenu_autograd::creator::rand::normal;
    use zenu_matrix::{device::Device, dim::DimTrait, operation::mul::matmul};
    use zenu_test::{assert_val_eq, run_test};

    use crate::Module;

    use super::Linear;

    fn with_bias<D: Device>() {
        let layer = Linear::<f32, D>::new(3, 2, true);
        let input = normal::<_, _, D>(0., 1., None, [5, 3]);
        let output = layer.call(input.clone());
        assert_eq!(output.get_data().shape().slice(), [5, 2]);

        let parameters = layer.parameters();
        let weight = parameters[0].clone();
        let bias = parameters[1].clone();

        let ans = matmul(&input.get_data().to_ref(), &weight.get_data().to_ref())
            + bias.get_data().to_ref();

        assert_val_eq!(output, ans, 1e-4);
    }
    run_test!(with_bias, with_bias_cpu, with_bias_gpu);

    fn without_bias<D: Device>() {
        let layer = Linear::<f32, D>::new(3, 2, false);
        let input = normal::<_, _, D>(0., 1., None, [5, 3]);
        let output = layer.call(input.clone());
        assert_eq!(output.get_data().shape().slice(), [5, 2]);

        let parameters = layer.parameters();
        let weight = parameters[0].clone();

        let ans = matmul(&input.get_data().to_ref(), &weight.get_data().to_ref());
        assert_val_eq!(output, ans, 1e-4);
    }
    run_test!(without_bias, without_bias_cpu, without_bias_gpu);
}
