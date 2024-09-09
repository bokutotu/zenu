use crate::{
    device::{nvidia::Nvidia, DeviceBase},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ptr, Ref},
    num::Num,
};

use super::{RNNBkwdDataOutput, RNNDescriptor, RNNOutput};

impl<T: Num> RNNDescriptor<T> {
    fn rnn_fwd_shape_check(&self, x: DimDyn, hx: Option<DimDyn>) {
        assert_eq!(x.len(), 3, "Input shape must be 3D");
        assert_eq!(x[1], self.get_batch_size(), "Batch size mismatch");
        assert_eq!(x[2], self.get_input_size(), "Input size mismatch");
        let num_layers = self.get_num_layers() * if self.get_is_bidirectional() { 2 } else { 1 };
        if let Some(hx) = hx {
            assert_eq!(hx[0], num_layers, "Number of layers mismatch");
            assert_eq!(hx[1], self.get_batch_size(), "Hidden size mismatch");
            assert_eq!(hx[2], self.get_hidden_size(), "Hidden size mismatch");
        }
    }

    #[expect(clippy::needless_pass_by_value)]
    pub fn rnn_fwd(
        &mut self,
        x: Matrix<Ref<&T>, DimDyn, Nvidia>,
        hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        weight: Matrix<Ref<&T>, DimDyn, Nvidia>,
        is_training: bool,
    ) -> RNNOutput<T> {
        self.rnn_fwd_shape_check(x.shape(), hx.as_ref().map(|hx| hx.to_ref().shape()));
        self.config_seq_length(is_training, x.shape()[0], x.shape()[1]);

        let hidden_size = self.get_hidden_size() * if self.get_is_bidirectional() { 2 } else { 1 };
        let mut y = Matrix::alloc([x.shape()[0], x.shape()[1], hidden_size]);

        let num_layers = self.get_num_layers() * if self.get_is_bidirectional() { 2 } else { 1 };

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
        RNNOutput { y, hy }
    }

    fn rnn_bkwd_data_shape_check(
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
        let hidden_size = self.get_hidden_size() * if self.get_is_bidirectional() { 2 } else { 1 };
        assert_eq!(y[2], hidden_size, "Hidden size mismatch");
        assert_eq!(y.slice(), dy.slice(), "Output and dy shape mismatch");
        assert_eq!(hx.is_some(), dhy.is_some(), "hx and dhy must be both None or both Some");

        if hx.is_none() && dhy.is_none() {
            return;
        }

        let hx = hx.unwrap();
        let dhy = dhy.unwrap();

        assert_eq!(hx.slice(), dhy.slice(), "hx and dhy shape mismatch");

        let num_layers = self.get_num_layers() * if self.get_is_bidirectional() { 2 } else { 1 };
        assert_eq!(hx[0], num_layers, "Number of layers mismatch");
        assert_eq!(hx[1], self.get_batch_size(), "Batch size mismatch");
        assert_eq!(hx[2], self.get_hidden_size(), "Hidden size mismatch");
    }

    #[expect(clippy::needless_pass_by_value, clippy::similar_names)]
    pub fn rnn_bkwd_data(
        &mut self,
        x_shape: DimDyn,
        y: Matrix<Ref<&T>, DimDyn, Nvidia>,
        dy: Matrix<Ref<&T>, DimDyn, Nvidia>,
        hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        dhy: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        weight: Matrix<Ref<&T>, DimDyn, Nvidia>,
    ) -> RNNBkwdDataOutput<T> {
        self.rnn_bkwd_data_shape_check(
            x_shape,
            y.shape(),
            dy.shape(),
            hx.as_ref().map(Matrix::shape),
            dhy.as_ref().map(Matrix::shape),
        );
        self.config_seq_length(true, x_shape[0], x_shape[1]);

        let mut dx = Matrix::alloc(x_shape);
        let mut dhx = {
            let d = self.desc.get_num_layers()
                * if self.desc.get_is_bidirectional() {
                    2
                } else {
                    1
                };
            Matrix::alloc([d, self.desc.get_hidden_size()])
        };

        self.bkwd_data(
            y.as_ptr(),
            dy.as_ptr(),
            dx.to_ref_mut().as_mut_ptr(),
            hx.map_or(std::ptr::null(), |hx| hx.as_ptr()),
            dhy.map_or(std::ptr::null(), |dhy| dhy.as_ptr()),
            dhx.to_ref_mut().as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            weight.as_ptr(),
        );
        RNNBkwdDataOutput { dx, dhx }
    }

    fn rnn_bkwd_weights_shape_check(&self, x: DimDyn, hx: Option<DimDyn>, y: DimDyn) {
        assert_eq!(x.len(), 3, "Input shape must be 3D");
        assert_eq!(x[1], self.get_batch_size(), "Batch size mismatch");
        assert_eq!(x[2], self.get_input_size(), "Input size mismatch");
        assert_eq!(y.len(), 3, "Output shape must be 3D");
        assert_eq!(y[1], self.get_batch_size(), "Batch size mismatch");
        let hidden_size = self.get_hidden_size() * if self.get_is_bidirectional() { 2 } else { 1 };
        assert_eq!(y[2], hidden_size, "Hidden size mismatch");

        let num_layers = self.get_num_layers() * if self.get_is_bidirectional() { 2 } else { 1 };
        if let Some(hx) = hx {
            assert_eq!(hx[0], num_layers, "Number of layers mismatch");
            assert_eq!(hx[1], self.get_batch_size(), "Batch size mismatch");
            assert_eq!(hx[2], self.get_hidden_size(), "Hidden size mismatch");
        }
    }

    #[expect(clippy::needless_pass_by_value, clippy::missing_panics_doc)]
    pub fn rnn_bkwd_weights(
        &mut self,
        x: Matrix<Ref<&T>, DimDyn, Nvidia>,
        hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        y: Matrix<Ref<&T>, DimDyn, Nvidia>,
    ) -> Matrix<Owned<T>, DimDyn, Nvidia> {
        self.rnn_bkwd_weights_shape_check(x.shape(), hx.as_ref().map(Matrix::shape), y.shape());
        self.config_seq_length(true, x.shape()[0], x.shape()[1]);

        let dweight = Nvidia::alloc(self.desc.get_weights_size()).unwrap();

        self.bkwd_weights(
            x.as_ptr(),
            hx.map_or(std::ptr::null(), |hx| hx.as_ptr()),
            y.as_ptr(),
            dweight.cast(),
        );
        let weight_size = self.get_weight_bytes() / std::mem::size_of::<T>();
        Matrix::new(
            Ptr::new(dweight.cast(), weight_size, 0),
            DimDyn::new(&[weight_size]),
            DimDyn::new(&[1]),
        )
    }
}
