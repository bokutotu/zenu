use std::collections::HashMap;
#[cfg(feature = "nvidia")]
use std::{cell::RefCell, rc::Rc};

use rand_distr::{Distribution, StandardNormal};
use zenu_autograd::{
    nn::rnns::lstm::naive::{lstm_naive, LSTMLayerWeights, LSTMWeights},
    Variable,
};
use zenu_matrix::{device::Device, num::Num};

#[cfg(feature = "nvidia")]
use zenu_autograd::nn::rnns::lstm::cudnn::lstm_cudnn;
#[cfg(feature = "nvidia")]
use zenu_matrix::{
    device::nvidia::Nvidia,
    nn::rnn::{RNNDescriptor, RNNWeights},
};

use crate::{Module, ModuleParameters, Parameters};

pub struct LSTM<T: Num, D: Device> {
    weights: Option<Vec<LSTMLayerWeights<T, D>>>,
    #[cfg(feature = "nvidia")]
    desc: Option<Rc<RefCell<RNNDescriptor<T>>>>,
    #[cfg(feature = "nvidia")]
    cudnn_weights: Option<Variable<T, Nvidia>>,
    #[cfg(feature = "nvidia")]
    is_cudnn: bool,
    is_bidirectional: bool,
    #[cfg(feature = "nvidia")]
    is_training: bool,
}

fn get_num_layers<T: Num, D: Device>(parameters: &HashMap<String, Variable<T, D>>) -> usize {
    let mut num_layers = 0;
    for key in parameters.keys() {
        if key.starts_with("lstm.") {
            let layer_num = key.split('.').nth(1).unwrap().parse::<usize>().unwrap();
            num_layers = num_layers.max(layer_num);
        }
    }
    num_layers + 1
}

fn is_bidirectional<T: Num, D: Device>(parameters: &HashMap<String, Variable<T, D>>) -> bool {
    for key in parameters.keys() {
        if key.starts_with("lstm.") && key.contains("reverse") {
            return true;
        }
    }
    false
}

fn get_nth_weights<T: Num, D: Device>(
    parameters: &HashMap<String, Variable<T, D>>,
    idx: usize,
    is_bidirectional: bool,
) -> LSTMLayerWeights<T, D> {
    let forward_weight_ih = parameters
        .get(&format!("lstm.{idx}.forward.weight_ih"))
        .unwrap()
        .clone();
    let forward_weight_hh = parameters
        .get(&format!("lstm.{idx}.forward.weight_hh"))
        .unwrap()
        .clone();
    let forward_bias_ih = parameters
        .get(&format!("lstm.{idx}.forward.bias_ih"))
        .unwrap()
        .clone();
    let forward_bias_hh = parameters
        .get(&format!("lstm.{idx}.forward.bias_hh"))
        .unwrap()
        .clone();

    let forward = LSTMWeights {
        weight_ih: forward_weight_ih,
        weight_hh: forward_weight_hh,
        bias_ih: forward_bias_ih,
        bias_hh: forward_bias_hh,
    };

    if is_bidirectional {
        let reverse_weight_ih = parameters
            .get(&format!("lstm.{idx}.reverse.weight_ih"))
            .unwrap()
            .clone();
        let reverse_weight_hh = parameters
            .get(&format!("lstm.{idx}.reverse.weight_hh"))
            .unwrap()
            .clone();
        let reverse_bias_ih = parameters
            .get(&format!("lstm.{idx}.reverse.bias_ih"))
            .unwrap()
            .clone();
        let reverse_bias_hh = parameters
            .get(&format!("lstm.{idx}.reverse.bias_hh"))
            .unwrap()
            .clone();

        let backward = LSTMWeights {
            weight_ih: reverse_weight_ih,
            weight_hh: reverse_weight_hh,
            bias_ih: reverse_bias_ih,
            bias_hh: reverse_bias_hh,
        };

        LSTMLayerWeights::new(forward, Some(backward))
    } else {
        LSTMLayerWeights::new(forward, None)
    }
}

impl<T: Num, D: Device> Parameters<T, D> for LSTM<T, D> {
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

            let forward_weight_ih = forward.weight_ih.clone();
            let forward_weight_hh = forward.weight_hh.clone();

            parameters.insert(
                format!("lstm.{idx}.forward.weight_ih"),
                forward_weight_ih.to(),
            );
            parameters.insert(
                format!("lstm.{idx}.forward.weight_hh"),
                forward_weight_hh.to(),
            );

            if self.is_bidirectional {
                let reverse = backward.unwrap();
                let reverse_weight_ih = reverse.weight_ih.clone();
                let reverse_weight_hh = reverse.weight_hh.clone();

                parameters.insert(
                    format!("lstm.{idx}.reverse.weight_ih"),
                    reverse_weight_ih.to(),
                );
                parameters.insert(
                    format!("lstm.{idx}.reverse.weight_hh"),
                    reverse_weight_hh.to(),
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

            let forward_bias_ih = forward.bias_ih.clone();
            let forward_bias_hh = forward.bias_hh.clone();

            parameters.insert(format!("lstm.{idx}.forward.bias_ih"), forward_bias_ih.to());
            parameters.insert(format!("lstm.{idx}.forward.bias_hh"), forward_bias_hh.to());

            if self.is_bidirectional {
                let reverse = backward.unwrap();
                let reverse_bias_ih = reverse.bias_ih.clone();
                let reverse_bias_hh = reverse.bias_hh.clone();

                parameters.insert(format!("lstm.{idx}.reverse.bias_ih"), reverse_bias_ih.to());
                parameters.insert(format!("lstm.{idx}.reverse.bias_hh"), reverse_bias_hh.to());
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

            let weights = lstm_weights_to_desc(weights, self.is_bidirectional);

            desc.borrow()
                .load_rnn_weights(cudnn_weights.get_as_mut().as_mut_ptr().cast(), weights)
                .unwrap();

            self.cudnn_weights = Some(cudnn_weights.clone());
            self.weights = None;
        }
    }
}

impl<T: Num, D: Device> LSTM<T, D> {
    #[cfg(feature = "nvidia")]
    fn cudnn_weights_to_layer_weights(&self) -> Vec<LSTMLayerWeights<T, D>> {
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
            .map(LSTMWeights::from)
            .collect::<Vec<LSTMWeights<T, D>>>();

        if self.is_bidirectional {
            let mut layer_weights = Vec::new();
            for i in 0..weights.len() / 2 {
                let forward = weights[i * 2].clone();
                let backward = weights[i * 2 + 1].clone();
                layer_weights.push(LSTMLayerWeights::new(forward, Some(backward)));
            }
            return layer_weights;
        }
        weights
            .into_iter()
            .map(|w| LSTMLayerWeights::new(w, None))
            .collect()
    }
}

pub struct LSTMInput<T: Num, D: Device> {
    pub x: Variable<T, D>,
    pub hx: Variable<T, D>,
    pub cx: Variable<T, D>,
}

impl<T: Num, D: Device> ModuleParameters<T, D> for LSTMInput<T, D> {}

#[cfg(feature = "nvidia")]
fn lstm_weights_to_desc<T: Num, D: Device>(
    weights: Vec<LSTMLayerWeights<T, D>>,
    is_bidirectional: bool,
) -> Vec<RNNWeights<T, D>> {
    let mut rnn_weights = Vec::new();

    for weight in weights {
        let forward_weights = weight.forward;
        let weight_ih = forward_weights.weight_ih.get_as_ref();
        let weight_hh = forward_weights.weight_hh.get_as_ref();
        let bias_ih = forward_weights.bias_ih.get_as_ref();
        let bias_hh = forward_weights.bias_hh.get_as_ref();

        let weights = RNNWeights::new(
            weight_ih.new_matrix(),
            weight_hh.new_matrix(),
            bias_ih.new_matrix(),
            bias_hh.new_matrix(),
        );

        rnn_weights.push(weights);

        if is_bidirectional {
            let backward_weights = weight.backward.unwrap();
            let weight_ih = backward_weights.weight_ih.get_as_ref();
            let weight_hh = backward_weights.weight_hh.get_as_ref();
            let bias_ih = backward_weights.bias_ih.get_as_ref();
            let bias_hh = backward_weights.bias_hh.get_as_ref();

            let weights = RNNWeights::new(
                weight_ih.new_matrix(),
                weight_hh.new_matrix(),
                bias_ih.new_matrix(),
                bias_hh.new_matrix(),
            );

            rnn_weights.push(weights);
        }
    }

    rnn_weights
}

impl<T: Num, D: Device> Module<T, D> for LSTM<T, D> {
    type Input = LSTMInput<T, D>;
    type Output = Variable<T, D>;
    fn call(&self, input: Self::Input) -> Self::Output {
        #[cfg(feature = "nvidia")]
        if self.is_cudnn {
            let desc = self.desc.as_ref().unwrap();
            let weights = self.cudnn_weights.as_ref().unwrap();

            let out = lstm_cudnn(
                desc.clone(),
                input.x.to(),
                Some(input.hx.to()),
                Some(input.cx.to()),
                weights.to(),
                self.is_training,
            );

            return out.to();
        }

        lstm_naive(
            input.x,
            input.hx,
            input.cx,
            self.weights.as_ref().unwrap(),
            self.is_bidirectional,
        )
    }
}

#[derive(Debug, Default)]
pub struct LSTMLayerBuilder<T: Num, D: Device> {
    is_cudnn: Option<bool>,
    is_bidirectional: Option<bool>,
    hidden_size: Option<usize>,
    num_layers: Option<usize>,
    input_size: Option<usize>,
    batch_size: Option<usize>,
    is_training: Option<bool>,
    _type: std::marker::PhantomData<(T, D)>,
}

impl<T: Num, D: Device> LSTMLayerBuilder<T, D> {
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
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    #[must_use]
    pub fn is_training(mut self, is_training: bool) -> Self {
        self.is_training = Some(is_training);
        self
    }

    fn init_weight(&self, idx: usize) -> LSTMLayerWeights<T, D>
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

        LSTMLayerWeights::init(
            input_size,
            self.hidden_size.unwrap(),
            self.is_bidirectional.unwrap(),
        )
    }

    fn init_weights(&self) -> Vec<LSTMLayerWeights<T, D>>
    where
        StandardNormal: Distribution<T>,
    {
        (0..self.num_layers.unwrap())
            .map(|idx| self.init_weight(idx))
            .collect()
    }

    #[cfg(feature = "nvidia")]
    fn init_cudnn_desc(&self) -> RNNDescriptor<T> {
        RNNDescriptor::<T>::lstm(
            self.is_bidirectional.unwrap(),
            0.0,
            self.input_size.unwrap(),
            self.hidden_size.unwrap(),
            self.num_layers.unwrap(),
            self.batch_size.unwrap(),
        )
    }

    #[cfg(feature = "nvidia")]
    fn load_cudnn_weights(
        &self,
        desc: &RNNDescriptor<T>,
        weights: Vec<LSTMLayerWeights<T, D>>,
    ) -> Variable<T, Nvidia> {
        use zenu_autograd::creator::alloc::alloc;
        let cudnn_weight_bytes = desc.get_weight_num_elems();
        let cudnn_weight: Variable<T, Nvidia> = alloc([cudnn_weight_bytes]);

        let weights = lstm_weights_to_desc(weights, self.is_bidirectional.unwrap_or(false));

        desc.load_rnn_weights(cudnn_weight.get_as_mut().as_mut_ptr().cast(), weights)
            .unwrap();

        cudnn_weight
    }

    #[expect(clippy::missing_panics_doc)]
    #[must_use]
    pub fn build(mut self) -> LSTM<T, D>
    where
        StandardNormal: Distribution<T>,
    {
        self.is_cudnn.get_or_insert(false);
        self.is_bidirectional.get_or_insert(false);
        self.is_training.get_or_insert(true);
        assert!(self.hidden_size.is_some(), "hidden_size is required");
        assert!(self.num_layers.is_some(), "num_layers is required");
        assert!(self.input_size.is_some(), "input_size is required");
        assert!(self.batch_size.is_some(), "batch_size is required");

        #[cfg(not(feature = "nvidia"))]
        assert!(
            !self.is_cudnn.unwrap(),
            "cuDNN is not enabled; please enable the 'nvidia' feature"
        );

        let weights = self.init_weights();
        #[cfg(feature = "nvidia")]
        if self.is_cudnn.unwrap() {
            let desc = self.init_cudnn_desc();
            let cudnn_weights = self.load_cudnn_weights(&desc, weights);
            let cudnn_weights = cudnn_weights.to::<Nvidia>();
            return LSTM {
                weights: None,
                desc: Some(Rc::new(RefCell::new(desc))),
                cudnn_weights: Some(cudnn_weights),
                is_cudnn: self.is_cudnn.unwrap(),
                is_bidirectional: self.is_bidirectional.unwrap(),
                is_training: self.is_training.unwrap(),
            };
        }
        LSTM {
            weights: Some(weights),
            #[cfg(feature = "nvidia")]
            desc: None,
            #[cfg(feature = "nvidia")]
            cudnn_weights: None,
            #[cfg(feature = "nvidia")]
            is_cudnn: self.is_cudnn.unwrap(),
            is_bidirectional: self.is_bidirectional.unwrap(),
            #[cfg(feature = "nvidia")]
            is_training: self.is_training.unwrap(),
        }
    }
}
