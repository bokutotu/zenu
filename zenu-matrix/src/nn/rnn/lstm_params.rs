use crate::{
    device::{nvidia::Nvidia, Device},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ref},
    num::Num,
};

use zenu_cuda::{
    cudnn::rnn::LSTMParams,
    runtime::{cuda_copy, ZenuCudaMemCopyKind},
};

use super::params::Params;

#[derive(Debug, Clone)]
pub struct LSTMWeightsMat<T: Num, D: Device> {
    input_gate_x: Matrix<Owned<T>, DimDyn, D>,
    input_gate_h: Matrix<Owned<T>, DimDyn, D>,
    forget_gate_x: Matrix<Owned<T>, DimDyn, D>,
    forget_gate_h: Matrix<Owned<T>, DimDyn, D>,
    cell_x: Matrix<Owned<T>, DimDyn, D>,
    cell_h: Matrix<Owned<T>, DimDyn, D>,
    output_gate_x: Matrix<Owned<T>, DimDyn, D>,
    output_gate_h: Matrix<Owned<T>, DimDyn, D>,
}

#[derive(Debug, Clone)]
pub struct LSTMOutput<T: Num> {
    pub y: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub hy: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub cy: Matrix<Owned<T>, DimDyn, Nvidia>,
}

#[derive(Debug, Clone)]
pub struct LSTMGrad<T: Num> {
    pub dx: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub dhx: Matrix<Owned<T>, DimDyn, Nvidia>,
    pub dcx: Matrix<Owned<T>, DimDyn, Nvidia>,
}

impl<T: Num, D: Device> LSTMWeightsMat<T, D> {
    #[expect(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        input_gate_x: Matrix<Owned<T>, DimDyn, D>,
        input_gate_h: Matrix<Owned<T>, DimDyn, D>,
        forget_gate_x: Matrix<Owned<T>, DimDyn, D>,
        forget_gate_h: Matrix<Owned<T>, DimDyn, D>,
        cell_x: Matrix<Owned<T>, DimDyn, D>,
        cell_h: Matrix<Owned<T>, DimDyn, D>,
        output_gate_x: Matrix<Owned<T>, DimDyn, D>,
        output_gate_h: Matrix<Owned<T>, DimDyn, D>,
    ) -> Self {
        Self {
            input_gate_x,
            input_gate_h,
            forget_gate_x,
            forget_gate_h,
            cell_x,
            cell_h,
            output_gate_x,
            output_gate_h,
        }
    }

    #[must_use]
    pub fn input_gate_x(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.input_gate_x.to_ref()
    }

    #[must_use]
    pub fn input_gate_h(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.input_gate_h.to_ref()
    }

    #[must_use]
    pub fn forget_gate_x(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.forget_gate_x.to_ref()
    }

    #[must_use]
    pub fn forget_gate_h(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.forget_gate_h.to_ref()
    }

    #[must_use]
    pub fn cell_x(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.cell_x.to_ref()
    }

    #[must_use]
    pub fn cell_h(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.cell_h.to_ref()
    }

    #[must_use]
    pub fn output_gate_x(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.output_gate_x.to_ref()
    }

    #[must_use]
    pub fn output_gate_h(&self) -> Matrix<Ref<&T>, DimDyn, D> {
        self.output_gate_h.to_ref()
    }
}

impl<T: Num, D: Device> Params for LSTMWeightsMat<T, D> {
    type Params = LSTMParams;
    #[expect(clippy::similar_names)]
    fn set_weight(&self, params: &LSTMParams) {
        let input_gates_x_ptr = params.input_gate_x.ptr.cast();
        let input_gates_h_ptr = params.input_gate_h.ptr.cast();
        let forget_gates_x_ptr = params.forget_gate_x.ptr.cast();
        let forget_gates_h_ptr = params.forget_gate_h.ptr.cast();
        let cell_x_ptr = params.cell_x.ptr.cast();
        let cell_h_ptr = params.cell_h.ptr.cast();
        let output_gates_x_ptr = params.output_gate_x.ptr.cast();
        let output_gates_h_ptr = params.output_gate_h.ptr.cast();

        let kind = if std::any::TypeId::of::<D>() == std::any::TypeId::of::<Nvidia>() {
            ZenuCudaMemCopyKind::HostToHost
        } else {
            ZenuCudaMemCopyKind::HostToDevice
        };

        cuda_copy(
            input_gates_x_ptr,
            self.input_gate_x.as_ptr(),
            self.input_gate_x.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            input_gates_h_ptr,
            self.input_gate_h.as_ptr(),
            self.input_gate_h.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            forget_gates_x_ptr,
            self.forget_gate_x.as_ptr(),
            self.forget_gate_x.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            forget_gates_h_ptr,
            self.forget_gate_h.as_ptr(),
            self.forget_gate_h.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            cell_x_ptr,
            self.cell_x.as_ptr(),
            self.cell_x.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            cell_h_ptr,
            self.cell_h.as_ptr(),
            self.cell_h.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            output_gates_x_ptr,
            self.output_gate_x.as_ptr(),
            self.output_gate_x.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            output_gates_h_ptr,
            self.output_gate_h.as_ptr(),
            self.output_gate_h.shape().num_elm(),
            kind,
        )
        .unwrap();
    }

    #[expect(clippy::similar_names)]
    fn load_from_params(&mut self, params: &LSTMParams) {
        let input_gates_x_ptr = params.input_gate_x.ptr as *const T;
        let input_gates_h_ptr = params.input_gate_h.ptr as *const T;
        let forget_gates_x_ptr = params.forget_gate_x.ptr as *const T;
        let forget_gates_h_ptr = params.forget_gate_h.ptr as *const T;
        let cell_x_ptr = params.cell_x.ptr as *const T;
        let cell_h_ptr = params.cell_h.ptr as *const T;
        let output_gates_x_ptr = params.output_gate_x.ptr as *const T;
        let output_gates_h_ptr = params.output_gate_h.ptr as *const T;

        let kind = if std::any::TypeId::of::<D>() == std::any::TypeId::of::<Nvidia>() {
            ZenuCudaMemCopyKind::HostToHost
        } else {
            ZenuCudaMemCopyKind::DeviceToHost
        };

        cuda_copy(
            self.input_gate_x.to_ref_mut().as_mut_ptr(),
            input_gates_x_ptr,
            self.input_gate_x.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            self.input_gate_h.to_ref_mut().as_mut_ptr(),
            input_gates_h_ptr,
            self.input_gate_h.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            self.forget_gate_x.to_ref_mut().as_mut_ptr(),
            forget_gates_x_ptr,
            self.forget_gate_x.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            self.forget_gate_h.to_ref_mut().as_mut_ptr(),
            forget_gates_h_ptr,
            self.forget_gate_h.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            self.cell_x.to_ref_mut().as_mut_ptr(),
            cell_x_ptr,
            self.cell_x.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            self.cell_h.to_ref_mut().as_mut_ptr(),
            cell_h_ptr,
            self.cell_h.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            self.output_gate_x.to_ref_mut().as_mut_ptr(),
            output_gates_x_ptr,
            self.output_gate_x.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            self.output_gate_h.to_ref_mut().as_mut_ptr(),
            output_gates_h_ptr,
            self.output_gate_h.shape().num_elm(),
            kind,
        )
        .unwrap();
    }
}
