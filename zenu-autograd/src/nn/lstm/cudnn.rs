use std::{cell::RefCell, rc::Rc};

use zenu_matrix::{nn::rnn::RNNDescriptor, num::Num};

use crate::{creator::alloc::alloc, Function, Variable, VariableWeak};

use zenu_matrix::device::nvidia::Nvidia;

struct CudnnLSTM<T: Num> {
    rnn_desc: Rc<RefCell<RNNDescriptor<T>>>,
    x: Variable<T, Nvidia>,
    weight: Variable<T, Nvidia>,
    hx: Option<Variable<T, Nvidia>>,
    cx: Option<Variable<T, Nvidia>>,
    hy: VariableWeak<T, Nvidia>,
    cy: VariableWeak<T, Nvidia>,
    y: VariableWeak<T, Nvidia>,
    is_training: bool,
}

impl<T: Num> Function<T, Nvidia> for CudnnLSTM<T> {
    fn forward(&self) {
        let mut rnn_desc = self.rnn_desc.borrow_mut();
        let hx = self.hx.as_ref().map(Variable::get_as_ref);
        let cx = self.cx.as_ref().map(Variable::get_as_ref);
        let output = rnn_desc.lstm_fwd(
            self.x.get_as_ref(),
            hx,
            cx,
            self.weight.get_as_ref(),
            self.is_training,
        );
        self.hy.upgrade().unwrap().swap_inner(output.hy);
        self.cy.upgrade().unwrap().swap_inner(output.cy);
        self.y.upgrade().unwrap().swap_inner(output.y);
    }

    #[expect(clippy::similar_names)]
    fn backward(&self) {
        assert!(
            self.is_training,
            "backward is not allowed when is_training is false"
        );

        let mut rnn_desc = self.rnn_desc.borrow_mut();

        let x_shape = self.x.get_shape();
        let y = self.y.upgrade().unwrap().get_as_ref();
        let dy = self.y.upgrade().unwrap().get_grad().unwrap().get_as_ref();
        let hx = self.hx.as_ref().map(Variable::get_as_ref);
        let cx = self.cx.as_ref().map(Variable::get_as_ref);
        let dhy = self
            .hy
            .upgrade()
            .map(|dhy| dhy.get_grad().unwrap().get_as_ref());
        let dcy = self
            .cy
            .upgrade()
            .map(|dcy| dcy.get_grad().unwrap().get_as_ref());
        let weight = self.weight.get_data().to_ref();
        let ddata = rnn_desc.lstm_bkwd_data(
            x_shape,
            y.clone(),
            dy,
            hx.clone(),
            cx.clone(),
            dhy,
            dcy,
            weight,
        );

        let dw = rnn_desc.lstm_bkwd_weights(self.x.get_as_ref(), hx.clone(), cx.clone(), y.clone());

        self.x.set_grad(Variable::new(ddata.dx));
        if let Some(hx) = self.hx.as_ref() {
            hx.set_grad(Variable::new(ddata.dhx));
        }
        if let Some(cx) = self.cx.as_ref() {
            cx.set_grad(Variable::new(ddata.dcx));
        }

        self.weight.set_grad(Variable::new(dw));
    }

    fn get_inputs(&self) -> Vec<Variable<T, Nvidia>> {
        let mut inputs = vec![self.x.clone()];
        if let Some(hx) = &self.hx {
            inputs.push(hx.clone());
        }
        if let Some(cx) = &self.cx {
            inputs.push(cx.clone());
        }
        inputs.push(self.weight.clone());
        inputs
    }
}

pub fn lstm_cudnn<T: Num>(
    rnn_desc: Rc<RefCell<RNNDescriptor<T>>>,
    x: Variable<T, Nvidia>,
    hx: Option<Variable<T, Nvidia>>,
    cx: Option<Variable<T, Nvidia>>,
    weight: Variable<T, Nvidia>,
    is_training: bool,
) -> Variable<T, Nvidia> {
    let num_layers = rnn_desc.borrow().get_num_layers();
    let hidden_size = rnn_desc.borrow().get_hidden_size();
    let x_shape = x.get_shape();
    let seq_len = x_shape[0];
    let batch_size = x_shape[1];
    let hy = alloc([num_layers, batch_size, hidden_size]);
    let cy = alloc([num_layers, batch_size, hidden_size]);
    let y = alloc([seq_len, batch_size, hidden_size]);
    let layer = CudnnLSTM {
        rnn_desc,
        x,
        hx,
        cx,
        hy: hy.clone().downgrade(),
        cy: cy.clone().downgrade(),
        y: y.clone().downgrade(),
        weight,
        is_training,
    };

    layer.forward();

    let layer = Rc::new(RefCell::new(Box::new(layer) as Box<dyn Function<T, Nvidia>>));
    hy.set_creator(layer.clone());
    cy.set_creator(layer.clone());
    y.set_creator(layer);

    // LSTMOutput { y, hy, cy }
    y
}

#[cfg(test)]
mod lstm_test {
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

    use super::lstm_cudnn;

    #[cfg(feature = "nvidia")]
    #[test]
    fn rnn() {
        let matrix_map =
            read_test_case_from_json_val!("../test_data_json/lstm_fwd_bkwd_small.json");

        let mut weights = Vec::new();
        let input_weight: Matrix<Owned<f32>, DimDyn, Cpu> =
            matrix_map.get("rnn.weight_ih_l0").unwrap().clone();
        let hidden_weight = matrix_map.get("rnn.weight_hh_l0").unwrap().clone();
        let input_bias = matrix_map.get("rnn.bias_ih_l0").unwrap().clone();
        let hidden_bias = matrix_map.get("rnn.bias_hh_l0").unwrap().clone();

        let rnn_weights = RNNWeightsMat::new(input_weight, hidden_weight, input_bias, hidden_bias);
        weights.push(rnn_weights);

        let input = matrix_map.get("input").unwrap().clone();
        let output = matrix_map.get("output").unwrap().clone();
        let input_size = input.shape()[2];
        let hidden_size = output.shape()[2];
        let batch_size = input.shape()[1];

        let rnn_desc =
            RNNDescriptor::<f32>::lstm(false, 0.0, input_size, hidden_size, 1, batch_size);

        let input = matrix_map.get("input").unwrap().clone();
        let input = Variable::new(input.to::<Nvidia>());

        let weight_num_elm = rnn_desc.get_weight_num_elems();
        let weight = Matrix::<Owned<f32>, DimDyn, Nvidia>::alloc([weight_num_elm]);
        let weight = Variable::new(weight);

        rnn_desc
            .load_rnn_weights(weight.get_as_mut().as_mut_ptr().cast(), weights)
            .unwrap();

        let rnn_desc = Rc::new(RefCell::new(rnn_desc));

        let result = lstm_cudnn(
            rnn_desc.clone(),
            input.clone(),
            None,
            None,
            weight.clone(),
            true,
        );

        result.backward();

        let expected_output = matrix_map.get("output").unwrap();

        assert_val_eq!(result, expected_output.clone().to::<Nvidia>(), 1e-5);
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
            g_input_bias,
            matrix_map
                .get("rnn.bias_ih_l0_grad")
                .unwrap()
                .clone()
                .to::<Cpu>(),
            1e-5
        );

        assert_mat_eq_epsilon!(
            g_hidden_bias,
            matrix_map
                .get("rnn.bias_hh_l0_grad")
                .unwrap()
                .clone()
                .to::<Cpu>(),
            1e-5
        );
    }
}
