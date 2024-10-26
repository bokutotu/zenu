use rand_distr::{Distribution, StandardNormal};
use zenu_autograd::{
    nn::rnns::{
        rnn::{
            cudnn::cudnn_rnn_fwd,
            naive::{rnn_relu, rnn_tanh},
            RNNOutput,
        },
        weights::{CellType, RNNCell},
    },
    Variable,
};
use zenu_matrix::{device::Device, num::Num};

use crate::{Module, ModuleParameters, Parameters};
#[cfg(feature = "nvidia")]
use zenu_matrix::device::nvidia::Nvidia;

use super::{
    builder::RNNSLayerBuilder,
    neo_struct::{Activation, NeoRNN},
};

pub struct RNNLayerInput<T: Num, D: Device> {
    pub x: Variable<T, D>,
    pub hx: Variable<T, D>,
}

impl<T: Num, D: Device> ModuleParameters<T, D> for RNNLayerInput<T, D> {}

impl<T: Num, D: Device> NeoRNN<T, D, RNNCell> {
    fn forward(&self, input: RNNLayerInput<T, D>) -> Variable<T, D> {
        #[cfg(feature = "nvidia")]
        if self.is_cudnn {
            let desc = self.desc.as_ref().unwrap();
            let weights = self.cudnn_weights.as_ref().unwrap();

            let out: RNNOutput<T, Nvidia> = cudnn_rnn_fwd(
                desc.clone(),
                input.x.to(),
                Some(input.hx.to()),
                weights.to(),
                self.is_training,
            );

            return out.y.to();
        }

        let activation = self.activation.unwrap();
        if activation == Activation::ReLU {
            rnn_relu(
                input.x,
                input.hx,
                self.weights.as_ref().unwrap(),
                self.is_bidirectional,
            )
        } else {
            rnn_tanh(
                input.x,
                input.hx,
                self.weights.as_ref().unwrap(),
                self.is_bidirectional,
            )
        }
    }
}

pub struct RNN<T: Num, D: Device>(NeoRNN<T, D, RNNCell>);

impl<T: Num, D: Device> Parameters<T, D> for RNN<T, D> {
    fn weights(&self) -> std::collections::HashMap<String, Variable<T, D>> {
        self.0.weights()
    }

    fn biases(&self) -> std::collections::HashMap<String, Variable<T, D>> {
        self.0.biases()
    }

    fn load_parameters(&mut self, parameters: std::collections::HashMap<String, Variable<T, D>>) {
        self.0.load_parameters(parameters)
    }
}

impl<T: Num, D: Device> Module<T, D> for RNN<T, D> {
    type Input = RNNLayerInput<T, D>;
    type Output = Variable<T, D>;

    fn call(&self, input: Self::Input) -> Self::Output {
        self.0.forward(input)
    }
}

impl<T: Num, D: Device> RNNSLayerBuilder<T, D, RNNCell> {
    pub fn build_rnn(self) -> RNN<T, D>
    where
        StandardNormal: Distribution<T>,
    {
        RNN(self.build_inner())
    }
}

pub type RNNBuilder<T, D> = RNNSLayerBuilder<T, D, RNNCell>;

#[cfg(test)]
mod rnn_layer_test {
    use zenu_autograd::creator::{rand::uniform, zeros::zeros};
    use zenu_matrix::{device::Device, dim::DimDyn};
    use zenu_test::{assert_val_eq, run_test};

    use crate::{Module, Parameters};

    use super::RNNBuilder;

    fn layer_save_load_test_not_cudnn<D: Device>() {
        let layer = RNNBuilder::<f32, D>::default()
            .hidden_size(10)
            .num_layers(2)
            .input_size(5)
            .batch_size(1)
            .build_rnn();

        let input = uniform(-1., 1., None, DimDyn::from([5, 1, 5]));
        let hidden = zeros([2, 1, 10]);

        let output = layer.call(super::RNNLayerInput {
            x: input.clone(),
            hx: hidden.clone(),
        });

        let parameters = layer.parameters();

        let new_layer = RNNBuilder::<f32, D>::default()
            .hidden_size(10)
            .num_layers(2)
            .input_size(5)
            .batch_size(1)
            .build_rnn();

        let new_layer_parameters = new_layer.parameters();

        for (key, value) in &parameters {
            new_layer_parameters
                .get(key)
                .unwrap()
                .get_as_mut()
                .copy_from(&value.get_as_ref());
        }

        let new_output = new_layer.call(super::RNNLayerInput {
            x: input,
            hx: hidden,
        });

        assert_val_eq!(output, new_output.get_as_ref(), 1e-4);
    }
    run_test!(
        layer_save_load_test_not_cudnn,
        layer_save_load_test_not_cudnn_cpu,
        layer_save_load_test_not_cudnn_gpu
    );

    #[cfg(feature = "nvidia")]
    #[test]
    fn layer_save_load_test_cudnn() {
        use zenu_matrix::device::nvidia::Nvidia;

        let layer = RNNBuilder::<f32, Nvidia>::default()
            .hidden_size(10)
            .num_layers(3)
            .input_size(5)
            .batch_size(5)
            .is_cudnn(true)
            .build_rnn();

        let mut new_layer = RNNBuilder::<f32, Nvidia>::default()
            .hidden_size(10)
            .num_layers(3)
            .input_size(5)
            .batch_size(5)
            .is_cudnn(true)
            .build_rnn();

        let layer_parameters = layer.parameters();

        new_layer.load_parameters(layer_parameters.clone());

        let new_layer_parameters = new_layer.parameters();

        for (key, value) in &layer_parameters {
            assert_val_eq!(
                value,
                new_layer_parameters.get(key).unwrap().get_as_ref(),
                1e-4
            );
        }
    }
}
