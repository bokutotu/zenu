use crate::{
    device::{nvidia::Nvidia, DeviceBase},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Ref},
    num::Num,
};

use super::{RNNBkwdDataOutput, RNNConfig, RNNOutput, RNNParameters};

fn rnn_fwd_shape_check<T: Num>(x: DimDyn, hx: Option<DimDyn>, config: &RNNConfig<T>) {
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
    match hx {
        Some(hx) => {
            if hx[0] != d {
                panic!("Number of layers mismatch");
            }
            if hx[1] != config.get_hidden_size() {
                panic!("Hidden size mismatch");
            }
        }
        None => {}
    }
}

pub fn rnn_fwd<T: Num>(
    x: Matrix<Ref<&T>, DimDyn, Nvidia>,
    hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
    is_training: bool,
    config: RNNConfig<T>,
    params: RNNParameters,
) -> RNNOutput<T> {
    rnn_fwd_shape_check(
        x.shape(),
        hx.as_ref().map(|hx| hx.to_ref().shape()),
        &config,
    );
    let rnn_exe = config.create_executor(is_training, x.shape()[1]);
    let reserve_size = rnn_exe.get_reserve_size();
    let workspace_size = rnn_exe.get_workspace_size();

    let reserve = Nvidia::alloc(reserve_size).unwrap();
    let workspace = Nvidia::alloc(workspace_size).unwrap();

    let mut y = Matrix::alloc([x.shape()[0], x.shape()[1], config.get_hidden_size()]);

    let d = config.get_num_layers() * if config.get_is_bidirectional() { 2 } else { 1 };

    let mut hy = Matrix::alloc([d, config.get_hidden_size()]);

    rnn_exe.fwd(
        x.as_ptr(),
        y.to_ref_mut().as_mut_ptr(),
        hx.map(|hx| hx.as_ptr()).unwrap_or(std::ptr::null()),
        hy.to_ref_mut().as_mut_ptr(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        params.weight as *mut _,
        workspace as *mut _,
        reserve as *mut _,
    );
    RNNOutput { y, hy }
}

fn rnn_bkwd_data_shape_check<T: Num>(
    x: DimDyn,
    y: DimDyn,
    dy: DimDyn,
    hx: Option<DimDyn>,
    dhy: Option<DimDyn>,
    config: &RNNConfig<T>,
) {
    if x.len() != 3 {
        panic!("Input shape must be 3D");
    }
    if x[1] != config.get_batch_size() {
        panic!("Batch size mismatch");
    }
    if x[2] != config.get_input_size() {
        panic!("Input size mismatch");
    }
    if y.len() != 3 {
        panic!("Output shape must be 3D");
    }
    if y[1] != config.get_batch_size() {
        panic!("Batch size mismatch");
    }
    if y[2] != config.get_hidden_size() {
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

    let d = config.get_num_layers() * if config.get_is_bidirectional() { 2 } else { 1 };
    if hx[0] != d {
        panic!("Number of layers mismatch");
    }
    if hx[1] != config.get_hidden_size() {
        panic!("Hidden size mismatch");
    }
}

pub fn rnn_bkwd_data<T: Num>(
    x: Matrix<Ref<&T>, DimDyn, Nvidia>,
    y: Matrix<Ref<&T>, DimDyn, Nvidia>,
    dy: Matrix<Ref<&T>, DimDyn, Nvidia>,
    hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
    dhy: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
    config: RNNConfig<T>,
    params: RNNParameters,
) -> RNNBkwdDataOutput<T> {
    rnn_bkwd_data_shape_check(
        x.shape(),
        y.shape(),
        dy.shape(),
        hx.as_ref().map(|hx| hx.shape()),
        dhy.as_ref().map(|dhy| dhy.shape()),
        &config,
    );
    let rnn_exe = config.create_executor(true, x.shape()[1]);
    let reserve_size = rnn_exe.get_reserve_size();
    let workspace_size = rnn_exe.get_workspace_size();

    let reserve = Nvidia::alloc(reserve_size).unwrap();
    let workspace = Nvidia::alloc(workspace_size).unwrap();

    let mut dx = Matrix::alloc(x.shape());
    let mut dhx = {
        let d = config.get_num_layers() * if config.get_is_bidirectional() { 2 } else { 1 };
        Matrix::alloc([d, config.get_hidden_size()])
    };

    rnn_exe.bkwd_data(
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
        workspace as *mut _,
        reserve as *mut _,
    );
    RNNBkwdDataOutput { dx, dhx }
}

fn rnn_bkwd_weights_shape_check<T: Num>(
    x: DimDyn,
    hx: Option<DimDyn>,
    y: DimDyn,
    config: &RNNConfig<T>,
) {
    if x.len() != 3 {
        panic!("Input shape must be 3D");
    }
    if x[1] != config.get_batch_size() {
        panic!("Batch size mismatch");
    }
    if x[2] != config.get_input_size() {
        panic!("Input size mismatch");
    }
    if y.len() != 3 {
        panic!("Output shape must be 3D");
    }
    if y[1] != config.get_batch_size() {
        panic!("Batch size mismatch");
    }
    if y[2] != config.get_hidden_size() {
        panic!("Hidden size mismatch");
    }

    let d = config.get_num_layers() * if config.get_is_bidirectional() { 2 } else { 1 };
    match hx {
        Some(hx) => {
            if hx[0] != d {
                panic!("Number of layers mismatch");
            }
            if hx[1] != config.get_hidden_size() {
                panic!("Hidden size mismatch");
            }
        }
        None => {}
    }
}

pub fn rnn_bkwd_weights<T: Num>(
    x: Matrix<Ref<&T>, DimDyn, Nvidia>,
    hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
    y: Matrix<Ref<&T>, DimDyn, Nvidia>,
    config: RNNConfig<T>,
) -> RNNParameters {
    rnn_bkwd_weights_shape_check(
        x.shape(),
        hx.as_ref().map(|hx| hx.shape()),
        y.shape(),
        &config,
    );
    let rnn_exe = config.create_executor(true, x.shape()[1]);
    let reserve_size = rnn_exe.get_reserve_size();
    let workspace_size = rnn_exe.get_workspace_size();

    let reserve = Nvidia::alloc(reserve_size).unwrap();
    let workspace = Nvidia::alloc(workspace_size).unwrap();

    let dweight = Nvidia::alloc(config.get_weight_bytes()).unwrap();

    rnn_exe.bkwd_weights(
        x.as_ptr(),
        hx.map(|hx| hx.as_ptr()).unwrap_or(std::ptr::null()),
        y.as_ptr(),
        dweight as *mut _,
        workspace as *mut _,
        reserve as *mut _,
    );
    RNNParameters { weight: dweight }
}
