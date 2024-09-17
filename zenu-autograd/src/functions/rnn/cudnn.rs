use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{nn::rnn::RNNDescriptor, num::Num};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

use zenu_matrix::device::nvidia::Nvidia;

use super::RNNOutput;

struct CudnnRNN<T: Num> {
    rnn_desc: Rc<RefCell<RNNDescriptor<T>>>,
    x: Variable<T, Nvidia>,
    weight: Variable<T, Nvidia>,
    hx: Option<Variable<T, Nvidia>>,
    hy: VariableWeak<T, Nvidia>,
    y: VariableWeak<T, Nvidia>,
    is_training: bool,
}

impl<T: Num> Function<T, Nvidia> for CudnnRNN<T> {
    fn forward(&self) {
        let mut rnn_desc = self.rnn_desc.borrow_mut();
        let hx = self.hx.as_ref().map(|hx| (*hx.get_data()).to_ref());
        let output = rnn_desc.rnn_fwd(
            self.x.get_as_ref(),
            hx,
            self.weight.get_as_ref(),
            self.is_training,
        );
        self.hy.upgrade().unwrap().swap_inner(output.hy);
        self.y.upgrade().unwrap().swap_inner(output.y);
    }

    fn backward(&self) {
        assert!(
            self.is_training,
            "backward is not allowed when is_training is false"
        );

        let mut rnn_desc = self.rnn_desc.borrow_mut();

        let x = self.x.get_as_ref();
        let y = self.y.upgrade().unwrap().get_as_ref();
        let dy = self.y.upgrade().unwrap().get_grad().unwrap().get_as_ref();
        let hx = self.hx.as_ref().map(|hx| (*hx.get_data()).to_ref());
        let dhy = self
            .hy
            .upgrade()
            .unwrap()
            .get_grad()
            .map(|dhy| dhy.get_as_ref());
        let weight = self.weight.get_data().to_ref();
        let ddata =
            rnn_desc.rnn_bkwd_data(self.x.get_shape(), y.clone(), dy, hx.clone(), dhy, weight);

        let dw = rnn_desc.rnn_bkwd_weights(x, hx, y);

        self.x.set_grad(Variable::new(ddata.dx));
        if let Some(hx) = self.hx.as_ref() {
            hx.set_grad(Variable::new(ddata.dhx));
        }

        self.weight.set_grad(Variable::new(dw));
    }

    fn get_inputs(&self) -> Vec<Variable<T, Nvidia>> {
        let mut inputs = vec![self.x.clone()];
        if let Some(hx) = &self.hx {
            inputs.push(hx.clone());
        }
        inputs.push(self.weight.clone());
        inputs
    }
}

pub fn cudnn_rnn_fwd<T: Num>(
    rnn_desc: Rc<RefCell<RNNDescriptor<T>>>,
    x: Variable<T, Nvidia>,
    hx: Option<Variable<T, Nvidia>>,
    weight: Variable<T, Nvidia>,
    is_training: bool,
) -> RNNOutput<T, Nvidia> {
    let num_layers = rnn_desc.borrow().get_num_layers();
    let hidden_size = rnn_desc.borrow().get_hidden_size();
    let x_shape = x.get_shape();
    let seq_len = x_shape[0];
    let batch_size = x_shape[1];
    let hy = alloc([num_layers, batch_size, hidden_size]);
    let y = alloc([seq_len, batch_size, hidden_size]);
    let layer = CudnnRNN {
        rnn_desc,
        x,
        hx,
        hy: hy.clone().downgrade(),
        y: y.clone().downgrade(),
        weight,
        is_training,
    };

    layer.forward();

    let layer = Rc::new(RefCell::new(Box::new(layer) as Box<dyn Function<T, Nvidia>>));
    hy.set_creator(layer.clone());
    y.set_creator(layer);

    RNNOutput { y, hy }
}

#[cfg(test)]
mod rnn_test {
    use std::{cell::RefCell, rc::Rc};

    use zenu_matrix::{
        device::{cpu::Cpu, nvidia::Nvidia},
        dim::DimDyn,
        matrix::{Matrix, Owned},
        nn::rnn::{RNNDescriptor, RNNWeightsMat},
    };

    use zenu_test::{
        assert_mat_eq_epsilon, assert_val_eq, assert_val_eq_grad, read_test_case_from_json_val,
    };

    use crate::Variable;

    use super::cudnn_rnn_fwd;

    #[cfg(feature = "nvidia")]
    #[test]
    fn rnn() {
        let matrix_map =
            read_test_case_from_json_val!("../test_data_json/rnn_fwd_bkwd_single_seq_len_1.json");

        let mut weights = Vec::new();
        let input_weight: Matrix<Owned<f32>, DimDyn, Cpu> =
            matrix_map.get("rnn.weight_ih_l0").unwrap().clone();
        let hidden_weight = matrix_map.get("rnn.weight_hh_l0").unwrap().clone();
        let input_bias = matrix_map.get("rnn.bias_ih_l0").unwrap().clone();
        let hidden_bias = matrix_map.get("rnn.bias_hh_l0").unwrap().clone();

        let rnn_weights = RNNWeightsMat::new(
            input_weight,
            hidden_weight,
            Some(input_bias),
            Some(hidden_bias),
        );
        weights.push(rnn_weights);

        let input = matrix_map.get("input").unwrap().clone();
        let output = matrix_map.get("output").unwrap().clone();
        let input_size = input.shape()[2];
        let hidden_size = output.shape()[2];
        let batch_size = input.shape()[1];

        let rnn_desc =
            RNNDescriptor::<f32>::new_rnn_relu(false, 0.0, input_size, hidden_size, 1, batch_size);

        let input = matrix_map.get("input").unwrap().clone();
        let input = Variable::new(input.to::<Nvidia>());

        let weight_num_elm = rnn_desc.get_weight_num_elems();
        let weight = Matrix::<Owned<f32>, DimDyn, Nvidia>::alloc([weight_num_elm]);
        let weight = Variable::new(weight);

        rnn_desc
            .load_rnn_weights(weight.get_as_mut().as_mut_ptr().cast(), weights)
            .unwrap();

        let rnn_desc = Rc::new(RefCell::new(rnn_desc));

        let result = cudnn_rnn_fwd(rnn_desc.clone(), input.clone(), None, weight.clone(), true);

        result.y.backward();

        let expected_output = matrix_map.get("output").unwrap();

        assert_val_eq!(result.y, expected_output.clone().to::<Nvidia>(), 1e-5);
        assert_val_eq_grad!(
            input.clone(),
            matrix_map.get("input_grad").unwrap().clone().to::<Nvidia>(),
            1e-5
        );

        let params = rnn_desc.borrow().store_rnn_weights::<Cpu>(
            weight
                .get_grad()
                .unwrap()
                .get_as_mut()
                .as_mut_ptr()
                .cast::<u8>(),
        );

        let g_input_weight = params[0].input_weight();
        let g_hidden_weight = params[0].hidden_weight();
        let g_input_bias = params[0].input_bias();
        let g_hidden_bias = params[0].hidden_bias();

        assert_mat_eq_epsilon!(
            g_input_weight,
            matrix_map
                .get("rnn.weight_ih_l0_grad")
                .unwrap()
                .clone()
                .to::<Cpu>(),
            1e-5
        );

        assert_mat_eq_epsilon!(
            g_hidden_weight,
            matrix_map
                .get("rnn.weight_hh_l0_grad")
                .unwrap()
                .clone()
                .to::<Cpu>(),
            1e-5
        );

        assert_mat_eq_epsilon!(
            g_input_bias.unwrap(),
            matrix_map
                .get("rnn.bias_ih_l0_grad")
                .unwrap()
                .clone()
                .to::<Cpu>(),
            1e-5
        );

        assert_mat_eq_epsilon!(
            g_hidden_bias.unwrap(),
            matrix_map
                .get("rnn.bias_hh_l0_grad")
                .unwrap()
                .clone()
                .to::<Cpu>(),
            1e-5
        );
    }
}
