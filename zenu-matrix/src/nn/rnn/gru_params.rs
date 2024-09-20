use zenu_cuda::{
    cudnn::rnn::GRUParams,
    runtime::{cuda_copy, ZenuCudaMemCopyKind},
};

use crate::{
    device::{nvidia::Nvidia, Device},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned},
    num::Num,
};

#[derive(Debug, Clone)]
pub struct GRUWeightsMat<T: Num, D: Device> {
    reset_gate_x: Matrix<Owned<T>, DimDyn, D>,
    reset_gate_h: Matrix<Owned<T>, DimDyn, D>,
    update_gate_x: Matrix<Owned<T>, DimDyn, D>,
    update_gate_h: Matrix<Owned<T>, DimDyn, D>,
    cell_x: Matrix<Owned<T>, DimDyn, D>,
    cell_h: Matrix<Owned<T>, DimDyn, D>,
}

impl<T: Num, D: Device> GRUWeightsMat<T, D> {
    #[must_use]
    pub fn new(
        reset_gate_x: Matrix<Owned<T>, DimDyn, D>,
        reset_gate_h: Matrix<Owned<T>, DimDyn, D>,
        update_gate_x: Matrix<Owned<T>, DimDyn, D>,
        update_gate_h: Matrix<Owned<T>, DimDyn, D>,
        cand_x: Matrix<Owned<T>, DimDyn, D>,
        cand_h: Matrix<Owned<T>, DimDyn, D>,
    ) -> Self {
        Self {
            reset_gate_x,
            reset_gate_h,
            update_gate_x,
            update_gate_h,
            cell_x: cand_x,
            cell_h: cand_h,
        }
    }

    #[must_use]
    pub fn reset_gate_x(&self) -> &Matrix<Owned<T>, DimDyn, D> {
        &self.reset_gate_x
    }

    #[must_use]
    pub fn reset_gate_h(&self) -> &Matrix<Owned<T>, DimDyn, D> {
        &self.reset_gate_h
    }

    #[must_use]
    pub fn update_gate_x(&self) -> &Matrix<Owned<T>, DimDyn, D> {
        &self.update_gate_x
    }

    #[must_use]
    pub fn update_gate_h(&self) -> &Matrix<Owned<T>, DimDyn, D> {
        &self.update_gate_h
    }

    #[must_use]
    pub fn cand_x(&self) -> &Matrix<Owned<T>, DimDyn, D> {
        &self.cell_x
    }

    #[must_use]
    pub fn cand_h(&self) -> &Matrix<Owned<T>, DimDyn, D> {
        &self.cell_h
    }

    #[expect(clippy::missing_panics_doc, clippy::similar_names)]
    pub fn set_weight(&self, params: &GRUParams) {
        let reset_gate_x_ptr = params.reset_gate_x.ptr.cast::<T>();
        let reset_gate_h_ptr = params.reset_gate_h.ptr.cast();
        let update_gate_x_ptr = params.update_gate_x.ptr.cast();
        let update_gate_h_ptr = params.update_gate_h.ptr.cast();
        let cell_x_ptr = params.cell_x.ptr.cast();
        let cell_h_ptr = params.cell_h.ptr.cast();

        let kind = if std::any::TypeId::of::<D>() == std::any::TypeId::of::<Nvidia>() {
            ZenuCudaMemCopyKind::HostToHost
        } else {
            ZenuCudaMemCopyKind::HostToDevice
        };

        cuda_copy(
            reset_gate_x_ptr,
            self.reset_gate_x.as_ptr(),
            self.reset_gate_x.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            reset_gate_h_ptr,
            self.reset_gate_h.as_ptr(),
            self.reset_gate_h.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            update_gate_x_ptr,
            self.update_gate_x.as_ptr(),
            self.update_gate_x.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            update_gate_h_ptr,
            self.update_gate_h.as_ptr(),
            self.update_gate_h.shape().num_elm(),
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
    }

    #[expect(clippy::missing_panics_doc, clippy::similar_names)]
    pub fn load_from_params(&mut self, params: &GRUParams) {
        let reset_gate_x_ptr = params.reset_gate_x.ptr as *const T;
        let reset_gate_h_ptr = params.reset_gate_h.ptr as *const T;
        let update_gate_x_ptr = params.update_gate_x.ptr as *const T;
        let update_gate_h_ptr = params.update_gate_h.ptr as *const T;
        let cell_x_ptr = params.cell_x.ptr as *const T;
        let cell_h_ptr = params.cell_h.ptr as *const T;

        let kind = if std::any::TypeId::of::<D>() == std::any::TypeId::of::<Nvidia>() {
            ZenuCudaMemCopyKind::HostToHost
        } else {
            ZenuCudaMemCopyKind::DeviceToHost
        };

        cuda_copy(
            self.reset_gate_x.to_ref_mut().as_mut_ptr(),
            reset_gate_x_ptr,
            self.reset_gate_x.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            self.reset_gate_h.to_ref_mut().as_mut_ptr(),
            reset_gate_h_ptr,
            self.reset_gate_h.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            self.update_gate_x.to_ref_mut().as_mut_ptr(),
            update_gate_x_ptr,
            self.update_gate_x.shape().num_elm(),
            kind,
        )
        .unwrap();
        cuda_copy(
            self.update_gate_h.to_ref_mut().as_mut_ptr(),
            update_gate_h_ptr,
            self.update_gate_h.shape().num_elm(),
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
    }
}
