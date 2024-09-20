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

use super::params::Params;

pub struct RNNOutput<T: Num> {
    pub y: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub hy: Matrix<Owned<T>, DimDyn, Nvidia>,
}

pub struct RNNBkwdDataOutput<T: Num> {
    pub dx: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub dhx: Matrix<Owned<T>, DimDyn, Nvidia>,
}

#[derive(Debug, Clone)]
pub struct RNNWeightsMat<T: Num, D: Device> {
    input_weight: Matrix<Owned<T>, DimDyn, D>,
    hidden_weight: Matrix<Owned<T>, DimDyn, D>,
    input_bias: Matrix<Owned<T>, DimDyn, D>,
    hidden_bias: Matrix<Owned<T>, DimDyn, D>,
}

impl<T: Num, D: Device> RNNWeightsMat<T, D> {
    #[must_use]
    pub fn new(
        input_weight: Matrix<Owned<T>, DimDyn, D>,
        hidden_weight: Matrix<Owned<T>, DimDyn, D>,
        input_bias: Matrix<Owned<T>, DimDyn, D>,
        hidden_bias: Matrix<Owned<T>, DimDyn, D>,
    ) -> Self {
        Self {
            input_weight,
            hidden_weight,
            input_bias,
            hidden_bias,
        }
    }

    #[must_use]
    pub fn input_weight(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.input_weight.to_ref()
    }

    #[must_use]
    pub fn hidden_weight(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.hidden_weight.to_ref()
    }

    #[must_use]
    pub fn input_bias(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.input_bias.to_ref()
    }

    #[must_use]
    pub fn hidden_bias(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.hidden_bias.to_ref()
    }
}

impl<T: Num, D: Device> Params for RNNWeightsMat<T, D> {
    type Params = RNNParams;
    fn set_weight(&self, params: &RNNParams) {
        let input_weight_ptr = params.input_weight.ptr.cast();
        let hidden_weight_ptr = params.hidden_weight.ptr.cast();
        let input_bias_ptr = params.input_bias.ptr.cast();
        let hidden_bias_ptr = params.hidden_bias.ptr.cast();

        let input_weight_numelm = self.input_weight().shape().num_elm();
        let hidden_weight_numelm = self.hidden_weight().shape().num_elm();
        let input_bias_numelm = self.input_bias().shape().num_elm();
        let hidden_bias_numelm = self.hidden_bias().shape().num_elm();

        let kind = if std::any::TypeId::of::<D>() == std::any::TypeId::of::<Nvidia>() {
            ZenuCudaMemCopyKind::HostToHost
        } else {
            ZenuCudaMemCopyKind::HostToDevice
        };

        cuda_copy(
            input_weight_ptr,
            self.input_weight.as_ptr(),
            input_weight_numelm,
            kind,
        )
        .unwrap();

        cuda_copy(
            hidden_weight_ptr,
            self.hidden_weight.as_ptr(),
            hidden_weight_numelm,
            kind,
        )
        .unwrap();

        cuda_copy(
            input_bias_ptr,
            self.input_bias.as_ptr(),
            input_bias_numelm,
            kind,
        )
        .unwrap();

        cuda_copy(
            hidden_bias_ptr,
            self.hidden_bias.as_ptr(),
            hidden_bias_numelm,
            kind,
        )
        .unwrap();
    }

    fn load_from_params(&mut self, params: &RNNParams) {
        let input_weight_ptr = params.input_weight.ptr as *const T;
        let hidden_weight_ptr = params.hidden_weight.ptr as *const T;
        let input_bias_ptr = params.input_bias.ptr as *const T;
        let hidden_bias_ptr = params.hidden_bias.ptr as *const T;

        let input_weight_numelm = self.input_weight().shape().num_elm();
        let hidden_weight_numelm = self.hidden_weight().shape().num_elm();
        let input_bias_numelm = self.input_bias().shape().num_elm();
        let hidden_bias_numelm = self.hidden_bias().shape().num_elm();

        let kind = if std::any::TypeId::of::<D>() == std::any::TypeId::of::<Nvidia>() {
            ZenuCudaMemCopyKind::HostToHost
        } else {
            ZenuCudaMemCopyKind::DeviceToHost
        };

        cuda_copy(
            self.input_weight.to_ref_mut().as_mut_ptr(),
            input_weight_ptr,
            input_weight_numelm,
            kind,
        )
        .unwrap();

        cuda_copy(
            self.hidden_weight.to_ref_mut().as_mut_ptr(),
            hidden_weight_ptr,
            hidden_weight_numelm,
            kind,
        )
        .unwrap();

        cuda_copy(
            self.input_bias.to_ref_mut().as_mut_ptr(),
            input_bias_ptr,
            input_bias_numelm,
            kind,
        )
        .unwrap();

        cuda_copy(
            self.hidden_bias.to_ref_mut().as_mut_ptr(),
            hidden_bias_ptr,
            hidden_bias_numelm,
            kind,
        )
        .unwrap();
    }
}
