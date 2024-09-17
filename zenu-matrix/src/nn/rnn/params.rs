use zenu_cuda::{
    cudnn::rnn::RNNParams,
    runtime::{cuda_copy, ZenuCudaMemCopyKind},
};

use crate::{
    device::{nvidia::Nvidia, Device, DeviceBase},
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

impl RNNParameters {
    #[expect(clippy::missing_panics_doc)]
    #[must_use]
    pub fn new(bytes: usize) -> Self {
        let weight = Nvidia::alloc(bytes).unwrap();
        Self { weight }
    }
}

impl Drop for RNNParameters {
    fn drop(&mut self) {
        Nvidia::drop_ptr(self.weight);
    }
}

#[derive(Debug, Clone)]
pub struct RNNWeightsMat<T: Num, D: Device> {
    pub input_weight: Matrix<Owned<T>, DimDyn, D>,
    pub hidden_weight: Matrix<Owned<T>, DimDyn, D>,
    pub input_bias: Option<Matrix<Owned<T>, DimDyn, D>>,
    pub hidden_bias: Option<Matrix<Owned<T>, DimDyn, D>>,
}

impl<T: Num, D: Device> RNNWeightsMat<T, D> {
    #[must_use]
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

    #[must_use]
    pub fn input_weight(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.input_weight.to_ref()
    }

    #[must_use]
    pub fn hidden_weight(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.hidden_weight.to_ref()
    }

    #[must_use]
    pub fn input_bias(&self) -> Option<Matrix<Ref<&T>, DimDyn, D>> {
        self.input_bias.as_ref().map(Matrix::to_ref)
    }

    #[must_use]
    pub fn hidden_bias(&self) -> Option<Matrix<Ref<&T>, DimDyn, D>> {
        self.hidden_bias.as_ref().map(Matrix::to_ref)
    }

    #[must_use]
    pub fn input_bias_mut(&mut self) -> Option<Matrix<Ref<&mut T>, DimDyn, D>> {
        self.input_bias.as_mut().map(Matrix::to_ref_mut)
    }

    #[must_use]
    pub fn hidden_bias_mut(&mut self) -> Option<Matrix<Ref<&mut T>, DimDyn, D>> {
        self.hidden_bias.as_mut().map(Matrix::to_ref_mut)
    }

    #[expect(clippy::missing_panics_doc)]
    pub fn set_weight(&self, params: &RNNParams) {
        let input_weight_ptr = params.input_weight.ptr.cast();
        let hidden_weight_ptr = params.hidden_weight.ptr.cast();
        let input_bias_ptr = params.input_bias.ptr.cast();
        let hidden_bias_ptr = params.hidden_bias.ptr.cast();

        let input_weight_numelm = self.input_weight().shape().num_elm();
        let hidden_weight_numelm = self.hidden_weight().shape().num_elm();
        let input_bias_numelm = self.input_bias.as_ref().map(|b| b.shape().num_elm());
        let hidden_bias_numelm = self.hidden_bias.as_ref().map(|b| b.shape().num_elm());

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

        if let Some(input_bias) = self.input_bias() {
            cuda_copy(
                input_bias_ptr,
                input_bias.as_ptr(),
                input_bias_numelm.unwrap(),
                kind,
            )
            .unwrap();
        }

        if let Some(hidden_bias) = self.hidden_bias() {
            cuda_copy(
                hidden_bias_ptr,
                hidden_bias.as_ptr(),
                hidden_bias_numelm.unwrap(),
                kind,
            )
            .unwrap();
        }
    }

    #[expect(clippy::missing_panics_doc)]
    pub fn load_from_params(&mut self, params: &RNNParams) {
        let input_weight_ptr = params.input_weight.ptr as *const T;
        let hidden_weight_ptr = params.hidden_weight.ptr as *const T;
        let input_bias_ptr = params.input_bias.ptr as *const T;
        let hidden_bias_ptr = params.hidden_bias.ptr as *const T;

        let input_weight_numelm = self.input_weight().shape().num_elm();
        let hidden_weight_numelm = self.hidden_weight().shape().num_elm();
        let input_bias_numelm = self.input_bias.as_ref().map(|b| b.shape().num_elm());
        let hidden_bias_numelm = self.hidden_bias.as_ref().map(|b| b.shape().num_elm());

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

        if let Some(input_bias) = self.input_bias_mut() {
            cuda_copy(
                input_bias.as_mut_ptr(),
                input_bias_ptr,
                input_bias_numelm.unwrap(),
                kind,
            )
            .unwrap();
        }

        if let Some(hidden_bias) = self.hidden_bias_mut() {
            cuda_copy(
                hidden_bias.as_mut_ptr(),
                hidden_bias_ptr,
                hidden_bias_numelm.unwrap(),
                kind,
            )
            .unwrap();
        }
    }
}
