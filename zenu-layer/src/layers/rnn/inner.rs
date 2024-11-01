use std::collections::HashMap;
#[cfg(feature = "nvidia")]
use std::{cell::RefCell, rc::Rc};

use zenu_autograd::{
    nn::rnns::weights::{CellType, RNNLayerWeights, RNNWeights},
    Variable,
};
#[cfg(feature = "nvidia")]
use zenu_matrix::{
    device::nvidia::Nvidia,
    nn::rnn::{RNNDescriptor, RNNWeights as RNNWeightsMat},
};

use zenu_matrix::{device::Device, num::Num};

use crate::Parameters;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    ReLU,
    Tanh,
}

#[expect(clippy::module_name_repetitions)]
pub struct RNNInner<T: Num, D: Device, C: CellType> {
    pub(super) weights: Option<Vec<RNNLayerWeights<T, D, C>>>,
    #[cfg(feature = "nvidia")]
    pub(super) desc: Option<Rc<RefCell<RNNDescriptor<T>>>>,
    #[cfg(feature = "nvidia")]
    pub(super) cudnn_weights: Option<Variable<T, Nvidia>>,
    #[cfg(feature = "nvidia")]
    pub(super) is_cudnn: bool,
    pub(super) is_bidirectional: bool,
    pub(super) activation: Option<Activation>,
    #[cfg(feature = "nvidia")]
    pub(super) is_training: bool,
}
fn get_num_layers<T: Num, D: Device, C: CellType>(
    parameters: &HashMap<String, Variable<T, D>>,
) -> usize {
    let mut num_layers = 0;
    for key in parameters.keys() {
        if key.starts_with(format!("{}.", C::name()).as_str()) {
            let layer_num = key.split('.').nth(1).unwrap().parse::<usize>().unwrap();
            num_layers = num_layers.max(layer_num);
        }
    }
    num_layers + 1
}

fn is_bidirectional<T: Num, D: Device, C: CellType>(
    parameters: &HashMap<String, Variable<T, D>>,
) -> bool {
    for key in parameters.keys() {
        if key.starts_with(format!("{}.", C::name()).as_str()) && key.contains("reverse") {
            return true;
        }
    }
    false
}

fn get_nth_weights<T: Num, D: Device, C: CellType>(
    parameters: &HashMap<String, Variable<T, D>>,
    idx: usize,
    is_bidirectional: bool,
) -> RNNLayerWeights<T, D, C> {
    let cell_name = C::name();
    let forward_input = parameters
        .get(&format!("{cell_name}.{idx}.forward.weight_input"))
        .unwrap()
        .clone();
    let forward_hidden = parameters
        .get(&format!("{cell_name}.{idx}.forward.weight_hidden"))
        .unwrap()
        .clone();
    let forward_bias_input = parameters
        .get(&format!("{cell_name}.{idx}.forward.bias_input"))
        .unwrap()
        .clone();
    let forward_bias_hidden = parameters
        .get(&format!("{cell_name}.{idx}.forward.bias_hidden"))
        .unwrap()
        .clone();

    let forward = RNNWeights::new(
        forward_input,
        forward_hidden,
        forward_bias_input,
        forward_bias_hidden,
    );

    if is_bidirectional {
        let reverse_input = parameters
            .get(&format!("{cell_name}.{idx}.reverse.weight_input"))
            .unwrap()
            .clone();
        let reverse_hidden = parameters
            .get(&format!("{cell_name}.{idx}.reverse.weight_hidden"))
            .unwrap()
            .clone();
        let reverse_bias_input = parameters
            .get(&format!("{cell_name}.{idx}.reverse.bias_input"))
            .unwrap()
            .clone();
        let reverse_bias_hidden = parameters
            .get(&format!("{cell_name}.{idx}.reverse.bias_hidden"))
            .unwrap()
            .clone();

        let backward = RNNWeights::new(
            reverse_input,
            reverse_hidden,
            reverse_bias_input,
            reverse_bias_hidden,
        );

        RNNLayerWeights::new(forward, Some(backward))
    } else {
        RNNLayerWeights::new(forward, None)
    }
}

impl<T: Num, D: Device, C: CellType> Parameters<T, D> for RNNInner<T, D, C> {
    fn weights(&self) -> HashMap<String, Variable<T, D>> {
        let cell_name = C::name();
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
                format!("{cell_name}.{idx}.forward.weight_input"),
                forward_input.to(),
            );
            parameters.insert(
                format!("{cell_name}.{idx}.forward.weight_hidden"),
                forward_hidden.to(),
            );

            if self.is_bidirectional {
                let reverse = backward.unwrap();
                let reverse_input = reverse.weight_input.clone();
                let reverse_hidden = reverse.weight_hidden.clone();

                parameters.insert(
                    format!("{cell_name}.{idx}.reverse.weight_input"),
                    reverse_input.to(),
                );
                parameters.insert(
                    format!("{cell_name}.{idx}.reverse.weight_hidden"),
                    reverse_hidden.to(),
                );
            }
        }

        parameters
    }

    fn biases(&self) -> HashMap<String, Variable<T, D>> {
        let cell_name = C::name();
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

            parameters.insert(
                format!("{cell_name}.{idx}.forward.bias_input"),
                forward_input.to(),
            );
            parameters.insert(
                format!("{cell_name}.{idx}.forward.bias_hidden"),
                forward_hidden.to(),
            );

            if self.is_bidirectional {
                let reverse = backward.unwrap();
                let reverse_input = reverse.bias_input.clone();
                let reverse_hidden = reverse.bias_hidden.clone();

                parameters.insert(
                    format!("{cell_name}.{idx}.reverse.bias_input"),
                    reverse_input.to(),
                );
                parameters.insert(
                    format!("{cell_name}.{idx}.reverse.bias_hidden"),
                    reverse_hidden.to(),
                );
            }
        }

        parameters
    }

    fn load_parameters(&mut self, parameters: HashMap<String, Variable<T, D>>) {
        let num_layers = get_num_layers::<T, D, C>(&parameters);
        let is_bidirectional = is_bidirectional::<T, D, C>(&parameters);

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

#[cfg(feature = "nvidia")]
pub(super) fn rnn_weights_to_desc<T: Num, D: Device, C: CellType>(
    weights: Vec<RNNLayerWeights<T, D, C>>,
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

impl<T: Num, D: Device, C: CellType> RNNInner<T, D, C> {
    #[cfg(feature = "nvidia")]
    fn cudnn_weights_to_layer_weights(&self) -> Vec<RNNLayerWeights<T, D, C>> {
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
            .collect::<Vec<_>>();

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
