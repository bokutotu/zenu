use crate::{
    device::nvidia::Nvidia,
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ref},
    num::Num,
};

use super::descriptor::RNNDescriptor;
use super::gru_params::{GRUGrad, GRUOutput};

impl<T: Num> RNNDescriptor<T> {
    fn gru_fwd_shape_check(&self, x: DimDyn, hx: Option<DimDyn>) {
        assert_eq!(x.len(), 3, "Input shape must be 3D");
        assert_eq!(x[1], self.get_batch_size(), "Batch size mismatch");
        assert_eq!(x[2], self.get_input_size(), "Input size mismatch");
        let output_size = self.get_output_size();
        let num_layers = self.get_output_num_layers();
        if let Some(hx) = hx {
            assert_eq!(hx[0], num_layers, "Number of layers mismatch");
            assert_eq!(hx[1], self.get_batch_size(), "Hidden size mismatch");
            assert_eq!(hx[2], output_size, "Hidden size mismatch");
        }
    }

    #[expect(clippy::needless_pass_by_value)]
    pub fn gru_fwd(
        &mut self,
        x: Matrix<Ref<&T>, DimDyn, Nvidia>,
        hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        weight: Matrix<Ref<&T>, DimDyn, Nvidia>,
        is_training: bool,
    ) -> GRUOutput<T> {
        self.gru_fwd_shape_check(x.shape(), hx.as_ref().map(|hx| hx.to_ref().shape()));

        self.config_seq_length(is_training, x.shape()[0], x.shape()[1]);

        let output_size = self.get_output_size();
        let mut y = Matrix::alloc([x.shape()[0], x.shape()[1], output_size]);

        let num_layers = self.get_output_num_layers();
        let mut hy = Matrix::zeros([num_layers, self.get_batch_size(), self.get_hidden_size()]);

        self.fwd(
            x.as_ptr(),
            y.to_ref_mut().as_mut_ptr(),
            hx.map_or(std::ptr::null(), |hx| hx.as_ptr()),
            hy.to_ref_mut().as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            weight.as_ptr(),
        );

        GRUOutput { y, hy }
    }

    fn gru_bkwd_data_shape_check(
        &self,
        x: DimDyn,
        y: DimDyn,
        dy: DimDyn,
        hx: Option<DimDyn>,
        dhy: Option<DimDyn>,
    ) {
        assert_eq!(x.len(), 3, "Input shape must be 3D");
        assert_eq!(x[1], self.get_batch_size(), "Batch size mismatch");
        assert_eq!(x[2], self.get_input_size(), "Input size mismatch");
        assert_eq!(y.len(), 3, "Output shape must be 3D");
        assert_eq!(y[1], self.get_batch_size(), "Batch size mismatch");
        assert_eq!(y[2], self.get_output_size(), "Output size mismatch");
        assert_eq!(dy.len(), 3, "dy shape must be 3D");
        assert_eq!(dy[1], self.get_batch_size(), "Batch size mismatch");
        assert_eq!(dy[2], self.get_output_size(), "dy size mismatch");
        let output_size = self.get_output_size();
        let num_layers = self.get_output_num_layers();
        if let Some(hx) = hx {
            assert_eq!(hx[0], num_layers, "Number of layers mismatch");
            assert_eq!(hx[1], self.get_batch_size(), "Hidden size mismatch");
            assert_eq!(hx[2], output_size, "Hidden size mismatch");
        }
        if let Some(dhy) = dhy {
            assert_eq!(dhy[0], num_layers, "Number of layers mismatch");
            assert_eq!(dhy[1], self.get_batch_size(), "Hidden size mismatch");
            assert_eq!(dhy[2], output_size, "Hidden size mismatch");
        }
    }

    #[expect(clippy::needless_pass_by_value, clippy::similar_names)]
    pub fn gru_bkwd_data(
        &mut self,
        x_shape: DimDyn,
        y: Matrix<Ref<&T>, DimDyn, Nvidia>,
        dy: Matrix<Ref<&T>, DimDyn, Nvidia>,
        hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        dhy: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        weight: Matrix<Ref<&T>, DimDyn, Nvidia>,
    ) -> GRUGrad<T> {
        self.gru_bkwd_data_shape_check(
            x_shape,
            y.shape(),
            dy.shape(),
            hx.as_ref().map(Matrix::shape),
            dhy.as_ref().map(Matrix::shape),
        );
        self.config_seq_length(true, x_shape[0], x_shape[1]);

        let mut dx = Matrix::alloc(x_shape);
        let mut dhx = Matrix::zeros([
            self.get_num_layers(),
            self.get_batch_size(),
            self.get_hidden_size(),
        ]);

        self.bkwd_data(
            y.as_ptr(),
            dy.as_ptr(),
            dx.to_ref_mut().as_mut_ptr(),
            hx.map_or(std::ptr::null(), |hx| hx.as_ptr()),
            dhy.map_or(std::ptr::null(), |dhy| dhy.as_ptr()),
            dhx.to_ref_mut().as_mut_ptr(),
            std::ptr::null(),     // GRU does not use cell states
            std::ptr::null(),     // GRU does not use cell states
            std::ptr::null_mut(), // GRU does not use cell states
            weight.as_ptr(),
        );

        GRUGrad { dx, dhx }
    }

    fn gru_bkwd_weights_shape_check(&self, x: DimDyn, hx: Option<DimDyn>, y: DimDyn) {
        assert_eq!(x.len(), 3, "Input shape must be 3D");
        assert_eq!(x[1], self.get_batch_size(), "Batch size mismatch");
        assert_eq!(x[2], self.get_input_size(), "Input size mismatch");
        let output_size = self.get_output_size();
        let num_layers = self.get_output_num_layers();
        if let Some(hx) = hx {
            assert_eq!(hx[0], num_layers, "Number of layers mismatch");
            assert_eq!(hx[1], self.get_batch_size(), "Hidden size mismatch");
            assert_eq!(hx[2], output_size, "Hidden size mismatch");
        }
        assert_eq!(y.len(), 3, "Output shape must be 3D");
        assert_eq!(y[1], self.get_batch_size(), "Batch size mismatch");
        assert_eq!(y[2], output_size, "Output size mismatch");
    }

    #[expect(clippy::needless_pass_by_value)]
    pub fn gru_bkwd_weights(
        &mut self,
        x: Matrix<Ref<&T>, DimDyn, Nvidia>,
        hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        y: Matrix<Ref<&T>, DimDyn, Nvidia>,
    ) -> Matrix<Owned<T>, DimDyn, Nvidia> {
        self.gru_bkwd_weights_shape_check(x.shape(), hx.as_ref().map(Matrix::shape), y.shape());
        self.config_seq_length(true, x.shape()[0], x.shape()[1]);

        let num_elm_weight = self.desc.get_weights_size() / std::mem::size_of::<T>();
        let mut dweight = Matrix::zeros([num_elm_weight]);

        self.bkwd_weights(
            x.as_ptr(),
            hx.map_or(std::ptr::null(), |hx| hx.as_ptr()),
            y.as_ptr(),
            dweight.to_ref_mut().as_mut_ptr(),
        );

        dweight
    }
}
