use crate::{
    device::nvidia::Nvidia,
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ref},
    num::Num,
};

use super::{
    descriptor::Descriptor,
    lstm_params::LSTMWeightsMat,
    lstm_params::{LSTMGrad, LSTMOutput},
};

impl<T: Num> Descriptor<T, LSTMWeightsMat<T, Nvidia>> {
    fn lstm_fwd_shape_check(&self, x: DimDyn, hx: Option<DimDyn>, cx: Option<DimDyn>) {
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
        if let Some(cx) = cx {
            assert_eq!(cx[0], num_layers, "Number of layers mismatch");
            assert_eq!(cx[1], self.get_batch_size(), "Hidden size mismatch");
            assert_eq!(cx[2], output_size, "Hidden size mismatch");
        }
    }

    #[expect(clippy::too_many_arguments, clippy::similar_names)]
    fn lstm_bkwd_data_shape_check(
        &self,
        x: DimDyn,
        y: DimDyn,
        dy: DimDyn,
        hx: Option<DimDyn>,
        cx: Option<DimDyn>,
        dhy: Option<DimDyn>,
        dcy: Option<DimDyn>,
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
        if let Some(cx) = cx {
            assert_eq!(cx[0], num_layers, "Number of layers mismatch");
            assert_eq!(cx[1], self.get_batch_size(), "Hidden size mismatch");
            assert_eq!(cx[2], output_size, "Hidden size mismatch");
        }
        if let Some(dhy) = dhy {
            assert_eq!(dhy[0], num_layers, "Number of layers mismatch");
            assert_eq!(dhy[1], self.get_batch_size(), "Hidden size mismatch");
            assert_eq!(dhy[2], output_size, "Hidden size mismatch");
        }
        if let Some(dcy) = dcy {
            assert_eq!(dcy[0], num_layers, "Number of layers mismatch");
            assert_eq!(dcy[1], self.get_batch_size(), "Hidden size mismatch");
            assert_eq!(dcy[2], output_size, "Hidden size mismatch");
        }
    }

    fn lstm_bkwd_weights_shape_check(
        &self,
        x: DimDyn,
        hx: Option<DimDyn>,
        cx: Option<DimDyn>,
        y: DimDyn,
    ) {
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
        if let Some(cx) = cx {
            assert_eq!(cx[0], num_layers, "Number of layers mismatch");
            assert_eq!(cx[1], self.get_batch_size(), "Hidden size mismatch");
            assert_eq!(cx[2], output_size, "Hidden size mismatch");
        }
        assert_eq!(y.len(), 3, "Output shape must be 3D");
        assert_eq!(y[1], self.get_batch_size(), "Batch size mismatch");
        assert_eq!(y[2], output_size, "Output size mismatch");
    }

    #[expect(clippy::needless_pass_by_value)]
    pub fn lstm_fwd(
        &mut self,
        x: Matrix<Ref<&T>, DimDyn, Nvidia>,
        hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        cx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        weight: Matrix<Ref<&T>, DimDyn, Nvidia>,
        is_training: bool,
    ) -> LSTMOutput<T> {
        self.lstm_fwd_shape_check(
            x.shape(),
            hx.as_ref().map(|hx| hx.to_ref().shape()),
            cx.as_ref().map(|cx| cx.to_ref().shape()),
        );

        self.config_seq_length(is_training, x.shape()[0], x.shape()[1]);

        let output_size = self.get_output_size();
        let mut y = Matrix::alloc([x.shape()[0], x.shape()[1], output_size]);
        let num_layers = self.get_output_num_layers();
        let mut hy = Matrix::zeros([num_layers, self.get_batch_size(), self.get_hidden_size()]);
        let mut cy = Matrix::zeros([num_layers, self.get_batch_size(), self.get_hidden_size()]);

        self.fwd(
            x.as_ptr(),
            y.to_ref_mut().as_mut_ptr(),
            hx.map_or(std::ptr::null(), |hx| hx.as_ptr()),
            hy.to_ref_mut().as_mut_ptr(),
            cx.map_or(std::ptr::null(), |cx| cx.as_ptr()),
            cy.to_ref_mut().as_mut_ptr(),
            weight.as_ptr(),
        );

        LSTMOutput { y, hy, cy }
    }

    #[expect(
        clippy::needless_pass_by_value,
        clippy::too_many_arguments,
        clippy::similar_names
    )]
    pub fn lstm_bkwd_data(
        &mut self,
        x: Matrix<Ref<&T>, DimDyn, Nvidia>,
        y: Matrix<Ref<&T>, DimDyn, Nvidia>,
        dy: Matrix<Ref<&T>, DimDyn, Nvidia>,
        hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        cx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        dhy: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        dcy: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        weight: Matrix<Ref<&T>, DimDyn, Nvidia>,
    ) -> LSTMGrad<T> {
        self.lstm_bkwd_data_shape_check(
            x.shape(),
            y.shape(),
            dy.shape(),
            hx.as_ref().map(|hx| hx.to_ref().shape()),
            cx.as_ref().map(|cx| cx.to_ref().shape()),
            dhy.as_ref().map(|dhy| dhy.to_ref().shape()),
            dcy.as_ref().map(|dcy| dcy.to_ref().shape()),
        );

        self.config_seq_length(true, x.shape()[0], x.shape()[1]);

        let output_size = self.get_output_size();
        let mut dx = Matrix::alloc([x.shape()[0], x.shape()[1], x.shape()[2]]);
        let mut dhx = Matrix::zeros([self.get_num_layers(), self.get_batch_size(), output_size]);
        let mut dcx = Matrix::zeros([self.get_num_layers(), self.get_batch_size(), output_size]);
        self.bkwd_data(
            y.as_ptr(),
            dy.as_ptr(),
            dx.to_ref_mut().as_mut_ptr(),
            hx.map_or(std::ptr::null(), |hx| hx.as_ptr()),
            dhy.map_or(std::ptr::null(), |dhy| dhy.as_ptr()),
            dhx.to_ref_mut().as_mut_ptr(),
            cx.map_or(std::ptr::null(), |cx| cx.as_ptr()),
            dcy.map_or(std::ptr::null(), |dcy| dcy.as_ptr()),
            dcx.to_ref_mut().as_mut_ptr(),
            weight.as_ptr(),
        );

        LSTMGrad { dx, dhx, dcx }
    }

    #[expect(clippy::needless_pass_by_value)]
    pub fn lstm_bkwd_weights(
        &mut self,
        x: Matrix<Ref<&T>, DimDyn, Nvidia>,
        hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        cx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        y: Matrix<Ref<&T>, DimDyn, Nvidia>,
    ) -> Matrix<Owned<T>, DimDyn, Nvidia> {
        self.lstm_bkwd_weights_shape_check(
            x.shape(),
            hx.as_ref().map(|hx| hx.to_ref().shape()),
            cx.as_ref().map(|cx| cx.to_ref().shape()),
            y.shape(),
        );

        let num_elm_weight = self.desc.get_weights_size() / std::mem::size_of::<T>();
        let mut dweight = Matrix::alloc([num_elm_weight]);

        self.bkwd_weights(
            x.as_ptr(),
            hx.map_or(std::ptr::null(), |hx| hx.as_ptr()),
            y.as_ptr(),
            dweight.to_ref_mut().as_mut_ptr(),
        );

        dweight
    }
}
