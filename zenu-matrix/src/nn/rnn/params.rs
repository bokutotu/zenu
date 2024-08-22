use zenu_cuda::{
    cudnn::rnn::RNNParams,
    runtime::{cuda_copy, ZenuCudaMemCopyKind},
};

use crate::{
    device::{nvidia::Nvidia, Device},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ref},
    num::Num,
};

pub struct RNNOutput<T: Num> {
    pub y: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub hy: Matrix<Owned<T>, DimDyn, Nvidia>,
}

pub struct RNNBkwdDataOutput<T: Num> {
    pub dx: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub dhx: Matrix<Owned<T>, DimDyn, Nvidia>,
}

pub struct RNNParameters {
    pub weight: *mut u8,
}

pub struct RNNWeights<T: Num, D: Device> {
    input_weight: Matrix<Owned<T>, DimDyn, D>,
    hidden_weight: Matrix<Owned<T>, DimDyn, D>,
    input_bias: Option<Matrix<Owned<T>, DimDyn, D>>,
    hidden_bias: Option<Matrix<Owned<T>, DimDyn, D>>,
}

impl<T: Num, D: Device> RNNWeights<T, D> {
    pub fn new(
        input_weight: Matrix<Owned<T>, DimDyn, D>,
        hidden_weight: Matrix<Owned<T>, DimDyn, D>,
        input_bias: Option<Matrix<Owned<T>, DimDyn, D>>,
        hidden_bias: Option<Matrix<Owned<T>, DimDyn, D>>,
    ) -> Self {
        Self {
            input_weight,
            hidden_weight,
            input_bias,
            hidden_bias,
        }
    }

    pub fn input_weight(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.input_weight.to_ref()
    }

    pub fn hidden_weight(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.hidden_weight.to_ref()
    }

    pub fn input_bias(&self) -> Option<Matrix<Ref<&T>, DimDyn, D>> {
        self.input_bias.as_ref().map(|b| b.to_ref())
    }

    pub fn hidden_bias(&self) -> Option<Matrix<Ref<&T>, DimDyn, D>> {
        self.hidden_bias.as_ref().map(|b| b.to_ref())
    }

    pub fn set_weight(&self, params: &RNNParams) {
        let input_weight_ptr = params.input_weight.ptr as *mut T;
        let hidden_weight_ptr = params.hidden_weight.ptr as *mut T;
        let input_bias_ptr = params.input_bias.ptr as *mut T;
        let hidden_bias_ptr = params.hidden_bias.ptr as *mut T;

        let input_weight_numelm = self.input_weight().shape().num_elm();
        let hidden_weight_numelm = self.hidden_weight().shape().num_elm();
        let input_bias_numelm = self.input_bias.as_ref().map(|b| b.shape().num_elm());
        let hidden_bias_numelm = self.hidden_bias.as_ref().map(|b| b.shape().num_elm());

        cuda_copy(
            input_weight_ptr,
            self.input_weight.as_ptr(),
            input_weight_numelm,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        cuda_copy(
            hidden_weight_ptr,
            self.hidden_weight.as_ptr(),
            hidden_weight_numelm,
            ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        if let Some(input_bias) = self.input_bias() {
            cuda_copy(
                input_bias_ptr,
                input_bias.as_ptr(),
                input_bias_numelm.unwrap(),
                ZenuCudaMemCopyKind::HostToDevice,
            )
            .unwrap();
        }

        if let Some(hidden_bias) = self.hidden_bias() {
            cuda_copy(
                hidden_bias_ptr,
                hidden_bias.as_ptr(),
                hidden_bias_numelm.unwrap(),
                ZenuCudaMemCopyKind::HostToDevice,
            )
            .unwrap();
        }
    }
}
