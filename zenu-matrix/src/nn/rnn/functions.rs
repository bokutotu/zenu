use crate::{
    device::{nvidia::Nvidia, DeviceBase},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Ref},
    num::Num,
};

use super::{RNNBkwdDataOutput, RNNDescriptor, RNNOutput, RNNParameters};

fn rnn_fwd_shape_check<T: Num>(x: DimDyn, hx: Option<DimDyn>, config: &RNNDescriptor<T>) {
    if x.len() != 3 {
        panic!("Input shape must be 3D");
    }
    if x[1] != config.get_batch_size() {
        panic!("Batch size mismatch");
    }
    if x[2] != config.get_input_size() {
        panic!("Input size mismatch");
    }
    let d = config.get_num_layers() * if config.get_is_bidirectional() { 2 } else { 1 };
    if let Some(hx) = hx {
        if hx[0] != d {
            panic!("Number of layers mismatch");
        }
        if hx[1] != config.get_hidden_size() {
            panic!("Hidden size mismatch");
        }
    }
}

pub fn rnn_fwd<T: Num>(
    x: Matrix<Ref<&T>, DimDyn, Nvidia>,
    hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
    is_training: bool,
    desc: &mut RNNDescriptor<T>,
    params: &RNNParameters,
) -> RNNOutput<T> {
    rnn_fwd_shape_check(x.shape(), hx.as_ref().map(|hx| hx.to_ref().shape()), desc);
    desc.config_seq_length(is_training, x.shape()[0]);

    let mut y = Matrix::alloc([x.shape()[0], x.shape()[1], desc.get_hidden_size()]);

    let d = desc.get_num_layers() * if desc.get_is_bidirectional() { 2 } else { 1 };

    let mut hy = Matrix::alloc([d, desc.get_hidden_size()]);

    desc.fwd(
        x.as_ptr(),
        y.to_ref_mut().as_mut_ptr(),
        hx.map(|hx| hx.as_ptr()).unwrap_or(std::ptr::null()),
        hy.to_ref_mut().as_mut_ptr(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        params.weight as *mut _,
    );
    RNNOutput { y, hy }
}

fn rnn_bkwd_data_shape_check<T: Num>(
    x: DimDyn,
    y: DimDyn,
    dy: DimDyn,
    hx: Option<DimDyn>,
    dhy: Option<DimDyn>,
    desc: &RNNDescriptor<T>,
) {
    if x.len() != 3 {
        panic!("Input shape must be 3D");
    }
    if x[1] != desc.get_batch_size() {
        panic!("Batch size mismatch");
    }
    if x[2] != desc.get_input_size() {
        panic!("Input size mismatch");
    }
    if y.len() != 3 {
        panic!("Output shape must be 3D");
    }
    if y[1] != desc.get_batch_size() {
        panic!("Batch size mismatch");
    }
    if y[2] != desc.get_hidden_size() {
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

    let d = desc.get_num_layers() * if desc.get_is_bidirectional() { 2 } else { 1 };
    if hx[0] != d {
        panic!("Number of layers mismatch");
    }
    if hx[1] != desc.get_hidden_size() {
        panic!("Hidden size mismatch");
    }
}

pub fn rnn_bkwd_data<T: Num>(
    x_shape: DimDyn,
    y: Matrix<Ref<&T>, DimDyn, Nvidia>,
    dy: Matrix<Ref<&T>, DimDyn, Nvidia>,
    hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
    dhy: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
    desc: &mut RNNDescriptor<T>,
    params: &RNNParameters,
) -> RNNBkwdDataOutput<T> {
    rnn_bkwd_data_shape_check(
        x_shape,
        y.shape(),
        dy.shape(),
        hx.as_ref().map(|hx| hx.shape()),
        dhy.as_ref().map(|dhy| dhy.shape()),
        desc,
    );
    desc.config_seq_length(true, x_shape[0]);

    let mut dx = Matrix::alloc(x_shape);
    let mut dhx = {
        let d = desc.desc.get_num_layers()
            * if desc.desc.get_is_bidirectional() {
                2
            } else {
                1
            };
        Matrix::alloc([d, desc.desc.get_hidden_size()])
    };

    desc.bkwd_data(
        y.as_ptr(),
        dy.as_ptr(),
        dx.to_ref_mut().as_mut_ptr(),
        hx.map(|hx| hx.as_ptr()).unwrap_or(std::ptr::null()),
        dhy.map(|dhy| dhy.as_ptr()).unwrap_or(std::ptr::null()),
        dhx.to_ref_mut().as_mut_ptr(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        params.weight as *mut _,
    );
    RNNBkwdDataOutput { dx, dhx }
}

fn rnn_bkwd_weights_shape_check<T: Num>(
    x: DimDyn,
    hx: Option<DimDyn>,
    y: DimDyn,
    desc: &RNNDescriptor<T>,
) {
    if x.len() != 3 {
        panic!("Input shape must be 3D");
    }
    if x[1] != desc.get_batch_size() {
        panic!("Batch size mismatch");
    }
    if x[2] != desc.get_input_size() {
        panic!("Input size mismatch");
    }
    if y.len() != 3 {
        panic!("Output shape must be 3D");
    }
    if y[1] != desc.get_batch_size() {
        panic!("Batch size mismatch");
    }
    if y[2] != desc.get_hidden_size() {
        panic!("Hidden size mismatch");
    }

    let d = desc.get_num_layers() * if desc.get_is_bidirectional() { 2 } else { 1 };
    if let Some(hx) = hx {
        if hx[0] != d {
            panic!("Number of layers mismatch");
        }
        if hx[1] != desc.get_hidden_size() {
            panic!("Hidden size mismatch");
        }
    }
}

pub fn rnn_bkwd_weights<T: Num>(
    x: Matrix<Ref<&T>, DimDyn, Nvidia>,
    hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
    y: Matrix<Ref<&T>, DimDyn, Nvidia>,
    desc: &mut RNNDescriptor<T>,
) -> RNNParameters {
    rnn_bkwd_weights_shape_check(
        x.shape(),
        hx.as_ref().map(|hx| hx.shape()),
        y.shape(),
        &desc,
    );
    desc.config_seq_length(true, x.shape()[1]);

    let dweight = Nvidia::alloc(desc.desc.get_weights_size()).unwrap();

    desc.bkwd_weights(
        x.as_ptr(),
        hx.map(|hx| hx.as_ptr()).unwrap_or(std::ptr::null()),
        y.as_ptr(),
        dweight as *mut _,
    );
    RNNParameters { weight: dweight }
}
