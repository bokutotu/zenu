use std::collections::HashMap;
#[cfg(feature = "nvidia")]
use std::{cell::RefCell, rc::Rc};

use rand_distr::{Distribution, StandardNormal};
use zenu_autograd::{
    nn::rnn::naive::{rnn_relu, rnn_tanh, RNNLayerWeights, RNNWeights},
    Variable,
};
use zenu_matrix::{device::Device, num::Num};

#[cfg(feature = "nvidia")]
use zenu_autograd::nn::rnn::{cudnn::cudnn_rnn_fwd, RNNOutput};
#[cfg(feature = "nvidia")]
use zenu_matrix::{
    device::nvidia::Nvidia,
    nn::rnn::{RNNDescriptor, RNNWeightsMat},
};

use crate::{Module, ModuleParameters, Parameters};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    ReLU,
    Tanh,
}

pub struct RNN<T: Num, D: Device> {
    weights: Option<Vec<RNNLayerWeights<T, D>>>,
    #[cfg(feature = "nvidia")]
    desc: Option<Rc<RefCell<RNNDescriptor<T>>>>,
    #[cfg(feature = "nvidia")]
    cudnn_weights: Option<Variable<T, Nvidia>>,
    #[cfg(feature = "nvidia")]
    is_cudnn: bool,
    is_bidirectional: bool,
    activation: Activation,
    #[cfg(feature = "nvidia")]
    is_training: bool,
}

fn get_num_layers<T: Num, D: Device>(parameters: &HashMap<String, Variable<T, D>>) -> usize {
    let mut num_layers = 0;
    for key in parameters.keys() {
        if key.starts_with("rnn.") {
            let layer_num = key.split('.').nth(1).unwrap().parse::<usize>().unwrap();
            num_layers = num_layers.max(layer_num);
        }
    }
    num_layers + 1
}

fn is_bidirectional<T: Num, D: Device>(parameters: &HashMap<String, Variable<T, D>>) -> bool {
    for key in parameters.keys() {
        if key.starts_with("rnn.") && key.contains("reverse") {
            return true;
        }
    }
    false
}

fn get_nth_weights<T: Num, D: Device>(
    parameters: &HashMap<String, Variable<T, D>>,
    idx: usize,
    is_bidirectional: bool,
) -> RNNLayerWeights<T, D> {
    let forward_input = parameters
        .get(&format!("rnn.{idx}.forward.weight_input"))
        .unwrap()
        .clone();
    let forward_hidden = parameters
        .get(&format!("rnn.{idx}.forward.weight_hidden"))
        .unwrap()
        .clone();
    let forward_bias_input = parameters
        .get(&format!("rnn.{idx}.forward.bias_input"))
        .unwrap()
        .clone();
    let forward_bias_hidden = parameters
        .get(&format!("rnn.{idx}.forward.bias_hidden"))
        .unwrap()
        .clone();

    let forward = RNNWeights {
        weight_input: forward_input,
        weight_hidden: forward_hidden,
        bias_input: forward_bias_input,
        bias_hidden: forward_bias_hidden,
    };

    if is_bidirectional {
        let reverse_input = parameters
            .get(&format!("rnn.{idx}.reverse.weight_input"))
            .unwrap()
            .clone();
        let reverse_hidden = parameters
            .get(&format!("rnn.{idx}.reverse.weight_hidden"))
            .unwrap()
            .clone();
        let reverse_bias_input = parameters
            .get(&format!("rnn.{idx}.reverse.bias_input"))
            .unwrap()
            .clone();
        let reverse_bias_hidden = parameters
            .get(&format!("rnn.{idx}.reverse.bias_hidden"))
            .unwrap()
            .clone();

        let backward = RNNWeights {
            weight_input: reverse_input,
            weight_hidden: reverse_hidden,
            bias_input: reverse_bias_input,
            bias_hidden: reverse_bias_hidden,
        };

        RNNLayerWeights::new(forward, Some(backward))
    } else {
        RNNLayerWeights::new(forward, None)
    }
}

impl<T: Num, D: Device> Parameters<T, D> for RNN<T, D> {
    fn weights(&self) -> HashMap<String, Variable<T, D>> {
        #[cfg(feature = "nvidia")]
        let weights = if self.is_cudnn {
            self.cudnn_weights_to_layer_weights()
        } else {
            self.weights.as_ref().unwrap().clone()
        };

        #[cfg(not(feature = "nvidia"))]
        let weights = self.weights.as_ref().unwrap().clone();

        let mut parameters = HashMap::new();

        for (idx, weight) in weights.iter().enumerate() {
            let forward = weight.forward.clone();
            let backward = weight.backward.clone();

            let forward_input = forward.weight_input.clone();
            let forward_hidden = forward.weight_hidden.clone();

            parameters.insert(
                format!("rnn.{idx}.forward.weight_input"),
                forward_input.to(),
            );
            parameters.insert(
                format!("rnn.{idx}.forward.weight_hidden"),
                forward_hidden.to(),
            );

            if self.is_bidirectional {
                let reverse = backward.unwrap();
                let reverse_input = reverse.weight_input.clone();
                let reverse_hidden = reverse.weight_hidden.clone();

                parameters.insert(
                    format!("rnn.{idx}.reverse.weight_input"),
                    reverse_input.to(),
                );
                parameters.insert(
                    format!("rnn.{idx}.reverse.weight_hidden"),
                    reverse_hidden.to(),
                );
            }
        }

        parameters
    }

    fn biases(&self) -> HashMap<String, Variable<T, D>> {
        #[cfg(feature = "nvidia")]
        let weights = if self.is_cudnn {
            self.cudnn_weights_to_layer_weights()
        } else {
            self.weights.as_ref().unwrap().clone()
        };

        #[cfg(not(feature = "nvidia"))]
        let weights = self.weights.as_ref().unwrap().clone();

        let mut parameters = HashMap::new();

        for (idx, weight) in weights.iter().enumerate() {
            let forward = weight.forward.clone();
            let backward = weight.backward.clone();

            let forward_input = forward.bias_input.clone();
            let forward_hidden = forward.bias_hidden.clone();

            parameters.insert(format!("rnn.{idx}.forward.bias_input"), forward_input.to());
            parameters.insert(
                format!("rnn.{idx}.forward.bias_hidden"),
                forward_hidden.to(),
            );

            if self.is_bidirectional {
                let reverse = backward.unwrap();
                let reverse_input = reverse.bias_input.clone();
                let reverse_hidden = reverse.bias_hidden.clone();

                parameters.insert(format!("rnn.{idx}.reverse.bias_input"), reverse_input.to());
                parameters.insert(
                    format!("rnn.{idx}.reverse.bias_hidden"),
                    reverse_hidden.to(),
                );
            }
        }

        parameters
    }

    fn load_parameters(&mut self, parameters: HashMap<String, Variable<T, D>>) {
        let num_layers = get_num_layers(&parameters);
        let is_bidirectional = is_bidirectional(&parameters);

        let mut weights = Vec::new();

        for idx in 0..num_layers {
            let weight = get_nth_weights(&parameters, idx, is_bidirectional);
            weights.push(weight);
        }

        self.weights = Some(weights.clone());

        #[cfg(feature = "nvidia")]
        if self.is_cudnn {
            let desc = self.desc.as_ref().unwrap();
            let cudnn_weights = self.cudnn_weights.as_ref().unwrap();

            let weights = rnn_weights_to_desc(weights, self.is_bidirectional);

            desc.borrow()
                .load_rnn_weights(cudnn_weights.get_as_mut().as_mut_ptr().cast(), weights)
                .unwrap();

            self.cudnn_weights = Some(cudnn_weights.clone());
            self.weights = None;
        }
    }
}

impl<T: Num, D: Device> RNN<T, D> {
    #[cfg(feature = "nvidia")]
    fn cudnn_weights_to_layer_weights(&self) -> Vec<RNNLayerWeights<T, D>> {
        use zenu_autograd::nn::rnn::naive::RNNWeights;

        let desc = self.desc.as_ref().unwrap().clone();
        let cudnn_weights_ptr = self
            .cudnn_weights
            .as_ref()
            .unwrap()
            .get_as_mut()
            .as_mut_ptr();
        let weights = desc
            .borrow()
            .store_rnn_weights::<D>(cudnn_weights_ptr.cast());

        let weights = weights
            .into_iter()
            .map(RNNWeights::from)
            .collect::<Vec<RNNWeights<T, D>>>();

        if self.is_bidirectional {
            let mut layer_weights = Vec::new();
            for i in 0..weights.len() / 2 {
                let forward = weights[i].clone();
                let backward = weights[i + 1].clone();
                layer_weights.push(RNNLayerWeights::new(forward, Some(backward)));
            }
            return layer_weights;
        }
        weights
            .into_iter()
            .map(|w| RNNLayerWeights::new(w, None))
            .collect()
    }
}

pub struct RNNLayerInput<T: Num, D: Device> {
    pub x: Variable<T, D>,
    pub hx: Variable<T, D>,
}

impl<T: Num, D: Device> ModuleParameters<T, D> for RNNLayerInput<T, D> {}

#[cfg(feature = "nvidia")]
fn rnn_weights_to_desc<T: Num, D: Device>(
    weights: Vec<RNNLayerWeights<T, D>>,
    is_bidirectional: bool,
) -> Vec<RNNWeightsMat<T, D>> {
    let mut rnn_weights = Vec::new();

    for weight in weights {
        let forwad_weights = weight.forward;
        let weight_input = forwad_weights.weight_input.get_as_ref();
        let weight_hidden = forwad_weights.weight_hidden.get_as_ref();
        let bias_input = forwad_weights.bias_input.get_as_ref();
        let bias_hidden = forwad_weights.bias_hidden.get_as_ref();

        let weights = RNNWeightsMat::new(
            weight_input.new_matrix(),
            weight_hidden.new_matrix(),
            bias_input.new_matrix(),
            bias_hidden.new_matrix(),
        );

        rnn_weights.push(weights);

        if is_bidirectional {
            let backward_weights = weight.backward.unwrap();
            let weight_input = backward_weights.weight_input.get_as_ref();
            let weight_hidden = backward_weights.weight_hidden.get_as_ref();
            let bias_input = backward_weights.bias_input.get_as_ref();
            let bias_hidden = backward_weights.bias_hidden.get_as_ref();

            let weights = RNNWeightsMat::new(
                weight_input.new_matrix(),
                weight_hidden.new_matrix(),
                bias_input.new_matrix(),
                bias_hidden.new_matrix(),
            );

            rnn_weights.push(weights);
        }
    }

    rnn_weights
}

impl<T: Num, D: Device> Module<T, D> for RNN<T, D> {
    type Input = RNNLayerInput<T, D>;
    type Output = Variable<T, D>;
    fn call(&self, input: Self::Input) -> Self::Output {
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

        if self.activation == Activation::ReLU {
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

#[derive(Debug, Default)]
pub struct RNNLayerBuilder<T: Num, D: Device> {
    is_cudnn: Option<bool>,
    is_bidirectional: Option<bool>,
    hidden_size: Option<usize>,
    num_layers: Option<usize>,
    input_size: Option<usize>,
    activation: Option<Activation>,
    batch_size: Option<usize>,
    is_training: Option<bool>,
    _type: std::marker::PhantomData<(T, D)>,
}

impl<T: Num, D: Device> RNNLayerBuilder<T, D> {
    #[must_use]
    pub fn is_cudnn(mut self, is_cudnn: bool) -> Self {
        self.is_cudnn = Some(is_cudnn);
        self
    }

    #[must_use]
    pub fn is_bidirectional(mut self, is_bidirectional: bool) -> Self {
        self.is_bidirectional = Some(is_bidirectional);
        self
    }

    #[must_use]
    pub fn hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = Some(hidden_size);
        self
    }

    #[must_use]
    pub fn num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = Some(num_layers);
        self
    }

    #[must_use]
    pub fn input_size(mut self, input_size: usize) -> Self {
        self.input_size = Some(input_size);
        self
    }

    #[must_use]
    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = Some(activation);
        self
    }

    #[must_use]
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    #[must_use]
    pub fn is_training(mut self, is_training: bool) -> Self {
        self.is_training = Some(is_training);
        self
    }

    fn init_weight(&self, idx: usize) -> RNNLayerWeights<T, D>
    where
        StandardNormal: Distribution<T>,
    {
        let input_size = if self.is_bidirectional.unwrap() {
            if idx == 0 {
                self.input_size.unwrap()
            } else {
                self.hidden_size.unwrap() * 2
            }
        } else if idx == 0 {
            self.input_size.unwrap()
        } else {
            self.hidden_size.unwrap()
        };

        RNNLayerWeights::init(
            input_size,
            self.hidden_size.unwrap(),
            self.is_bidirectional.unwrap(),
        )
    }

    fn init_weights(&self) -> Vec<RNNLayerWeights<T, D>>
    where
        StandardNormal: Distribution<T>,
    {
        (0..self.num_layers.unwrap())
            .map(|idx| self.init_weight(idx))
            .collect()
    }

    #[cfg(feature = "nvidia")]
    fn init_cudnn_desc(&self) -> RNNDescriptor<T> {
        if self.activation.unwrap() == Activation::ReLU {
            RNNDescriptor::<T>::new_rnn_relu(
                self.is_bidirectional.unwrap(),
                0.0,
                self.input_size.unwrap(),
                self.hidden_size.unwrap(),
                self.num_layers.unwrap(),
                self.batch_size.unwrap(),
            )
        } else {
            RNNDescriptor::<T>::new_rnn_tanh(
                self.is_bidirectional.unwrap(),
                0.0,
                self.input_size.unwrap(),
                self.hidden_size.unwrap(),
                self.num_layers.unwrap(),
                self.batch_size.unwrap(),
            )
        }
    }

    #[cfg(feature = "nvidia")]
    fn load_cudnn_weights(
        &self,
        desc: &RNNDescriptor<T>,
        weights: Vec<RNNLayerWeights<T, D>>,
    ) -> Variable<T, Nvidia> {
        use zenu_autograd::creator::alloc::alloc;
        let cudnn_weight_bytes = desc.get_weight_num_elems();
        let cudnn_weight: Variable<T, Nvidia> = alloc([cudnn_weight_bytes]);

        let weights = rnn_weights_to_desc(weights, self.is_bidirectional.unwrap_or(false));

        desc.load_rnn_weights(cudnn_weight.get_as_mut().as_mut_ptr().cast(), weights)
            .unwrap();

        cudnn_weight
    }

    #[expect(clippy::missing_panics_doc)]
    #[must_use]
    pub fn build(mut self) -> RNN<T, D>
    where
        StandardNormal: Distribution<T>,
    {
        self.is_cudnn.get_or_insert(false);
        self.is_bidirectional.get_or_insert(false);
        self.activation.get_or_insert(Activation::ReLU);
        self.is_training.get_or_insert(true);
        assert!(self.hidden_size.is_some(), "hidden_size is required");
        assert!(self.num_layers.is_some(), "num_layers is required");
        assert!(self.input_size.is_some(), "input_size is required");
        assert!(self.batch_size.is_some(), "batch_size is required");

        #[cfg(not(feature = "nvidia"))]
        assert!(
            !self.is_cudnn.unwrap(),
            "cudnn is not enabled, please enable cudnn feature"
        );

        let weights = self.init_weights();
        #[cfg(feature = "nvidia")]
        if self.is_cudnn.unwrap() {
            let desc = self.init_cudnn_desc();
            let cudnn_weights = self.load_cudnn_weights(&desc, weights);
            let cudnn_weights = cudnn_weights.to::<Nvidia>();
            return RNN {
                weights: None,
                desc: Some(Rc::new(RefCell::new(desc))),
                cudnn_weights: Some(cudnn_weights),
                is_cudnn: self.is_cudnn.unwrap(),
                is_bidirectional: self.is_bidirectional.unwrap(),
                activation: self.activation.unwrap(),
                is_training: self.is_training.unwrap(),
            };
        }
        RNN {
            weights: Some(weights),
            #[cfg(feature = "nvidia")]
            desc: None,
            #[cfg(feature = "nvidia")]
            cudnn_weights: None,
            #[cfg(feature = "nvidia")]
            is_cudnn: self.is_cudnn.unwrap(),
            is_bidirectional: self.is_bidirectional.unwrap(),
            activation: self.activation.unwrap(),
            #[cfg(feature = "nvidia")]
            is_training: self.is_training.unwrap(),
        }
    }
}

#[cfg(test)]
mod rnn_layer_test {
    use zenu_autograd::creator::{rand::uniform, zeros::zeros};
    use zenu_matrix::{device::Device, dim::DimDyn};
    use zenu_test::{assert_val_eq, run_test};

    use crate::{Module, Parameters};

    use super::RNNLayerBuilder;

    fn layer_save_load_test_not_cudnn<D: Device>() {
        let layer = RNNLayerBuilder::<f32, D>::default()
            .hidden_size(10)
            .num_layers(2)
            .input_size(5)
            .batch_size(1)
            .build();

        let input = uniform(-1., 1., None, DimDyn::from([5, 1, 5]));
        let hidden = zeros([2, 1, 10]);

        let output = layer.call(super::RNNLayerInput {
            x: input.clone(),
            hx: hidden.clone(),
        });

        let parameters = layer.parameters();

        let new_layer = RNNLayerBuilder::<f32, D>::default()
            .hidden_size(10)
            .num_layers(2)
            .input_size(5)
            .batch_size(1)
            .build();

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

        let layer = RNNLayerBuilder::<f32, Nvidia>::default()
            .hidden_size(10)
            .num_layers(3)
            .input_size(5)
            .batch_size(5)
            .is_cudnn(true)
            .build();

        let mut new_layer = RNNLayerBuilder::<f32, Nvidia>::default()
            .hidden_size(10)
            .num_layers(3)
            .input_size(5)
            .batch_size(5)
            .is_cudnn(true)
            .build();

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
