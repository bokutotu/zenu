use crate::{
    device::{nvidia::Nvidia, DeviceBase},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Owned, Ptr, Ref},
    num::Num,
};

use super::{RNNBkwdDataOutput, RNNDescriptor, RNNOutput};

impl<T: Num> RNNDescriptor<T> {
    fn rnn_fwd_shape_check(&self, x: DimDyn, hx: Option<DimDyn>) {
        if x.len() != 3 {
            panic!("Input shape must be 3D");
        }
        if x[1] != self.get_batch_size() {
            panic!("Batch size mismatch");
        }
        if x[2] != self.get_input_size() {
            panic!("Input size mismatch");
        }
        let d = self.get_num_layers() * if self.get_is_bidirectional() { 2 } else { 1 };
        if let Some(hx) = hx {
            if hx[0] != d {
                panic!("Number of layers mismatch");
            }
            if hx[1] != self.get_hidden_size() {
                panic!("Hidden size mismatch");
            }
        }
    }

    pub fn rnn_fwd(
        &mut self,
        x: Matrix<Ref<&T>, DimDyn, Nvidia>,
        hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        weight: Matrix<Ref<&T>, DimDyn, Nvidia>,
        is_training: bool,
    ) -> RNNOutput<T> {
        self.rnn_fwd_shape_check(x.shape(), hx.as_ref().map(|hx| hx.to_ref().shape()));
        self.config_seq_length(is_training, x.shape()[0]);

        let mut y = Matrix::alloc([x.shape()[0], x.shape()[1], self.get_hidden_size()]);

        let d = self.get_num_layers() * if self.get_is_bidirectional() { 2 } else { 1 };

        let mut hy = Matrix::alloc([d, self.get_hidden_size()]);

        self.fwd(
            x.as_ptr(),
            y.to_ref_mut().as_mut_ptr(),
            hx.map(|hx| hx.as_ptr()).unwrap_or(std::ptr::null()),
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
        if x.len() != 3 {
            panic!("Input shape must be 3D");
        }
        if x[1] != self.get_batch_size() {
            panic!("Batch size mismatch");
        }
        if x[2] != self.get_input_size() {
            panic!("Input size mismatch");
        }
        if y.len() != 3 {
            panic!("Output shape must be 3D");
        }
        if y[1] != self.get_batch_size() {
            panic!("Batch size mismatch");
        }
        if y[2] != self.get_hidden_size() {
            panic!("Hidden size mismatch");
        }
        if y.slice() != dy.slice() {
            panic!("Output and dy shape mismatch");
        }

        if hx.is_some() != dhy.is_some() {
            panic!("hx and dhy must be both None or both Some");
        }

        if hx.is_none() && dhy.is_none() {
            return;
        }

        let hx = hx.unwrap();
        let dhy = dhy.unwrap();

        if hx.slice() != dhy.slice() {
            panic!("hx and dhy shape mismatch");
        }

        let d = self.get_num_layers() * if self.get_is_bidirectional() { 2 } else { 1 };
        if hx[0] != d {
            panic!("Number of layers mismatch");
        }
        if hx[1] != self.get_hidden_size() {
            panic!("Hidden size mismatch");
        }
    }

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
            hx.as_ref().map(|hx| hx.shape()),
            dhy.as_ref().map(|dhy| dhy.shape()),
        );
        self.config_seq_length(true, x_shape[0]);

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
            hx.map(|hx| hx.as_ptr()).unwrap_or(std::ptr::null()),
            dhy.map(|dhy| dhy.as_ptr()).unwrap_or(std::ptr::null()),
            dhx.to_ref_mut().as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            weight.as_ptr(),
        );
        RNNBkwdDataOutput { dx, dhx }
    }

    fn rnn_bkwd_weights_shape_check(&self, x: DimDyn, hx: Option<DimDyn>, y: DimDyn) {
        if x.len() != 3 {
            panic!("Input shape must be 3D");
        }
        if x[1] != self.get_batch_size() {
            panic!("Batch size mismatch");
        }
        if x[2] != self.get_input_size() {
            panic!("Input size mismatch");
        }
        if y.len() != 3 {
            panic!("Output shape must be 3D");
        }
        if y[1] != self.get_batch_size() {
            panic!("Batch size mismatch");
        }
        if y[2] != self.get_hidden_size() {
            panic!("Hidden size mismatch");
        }

        let d = self.get_num_layers() * if self.get_is_bidirectional() { 2 } else { 1 };
        if let Some(hx) = hx {
            if hx[0] != d {
                panic!("Number of layers mismatch");
            }
            if hx[1] != self.get_hidden_size() {
                panic!("Hidden size mismatch");
            }
        }
    }

    pub fn rnn_bkwd_weights(
        &mut self,
        x: Matrix<Ref<&T>, DimDyn, Nvidia>,
        hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
        y: Matrix<Ref<&T>, DimDyn, Nvidia>,
    ) -> Matrix<Owned<T>, DimDyn, Nvidia> {
        self.rnn_bkwd_weights_shape_check(x.shape(), hx.as_ref().map(|hx| hx.shape()), y.shape());
        self.config_seq_length(true, x.shape()[0]);

        let dweight = Nvidia::alloc(self.desc.get_weights_size()).unwrap();

        self.bkwd_weights(
            x.as_ptr(),
            hx.map(|hx| hx.as_ptr()).unwrap_or(std::ptr::null()),
            y.as_ptr(),
            dweight as *mut _,
        );
        let weight_size = self.get_weight_bytes() / std::mem::size_of::<T>();
        Matrix::new(
            Ptr::new(dweight as *mut T, weight_size, 0),
            DimDyn::new(&[weight_size]),
            DimDyn::new(&[1]),
        )
    }
}
