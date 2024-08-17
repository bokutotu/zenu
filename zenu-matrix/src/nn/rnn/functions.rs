use crate::{
    device::{nvidia::Nvidia, DeviceBase},
    dim::{DimDyn, DimTrait},
    matrix::{Matrix, Ref},
    num::Num,
};

use super::{RNNConfig, RNNOutput, RNNParameters};

fn rnn_fwd_shape_check(x: DimDyn, hx: Option<DimDyn>, config: &RNNConfig) {
    if x.len() != 3 {
        panic!("Input shape must be 3D");
    }
    if x[1] != config.get_batch_size() {
        panic!("Batch size mismatch");
    }
    if x[2] != config.get_input_size() {
        panic!("Input size mismatch");
    }
    let d = cofnig.get_num_layers() * if config.is_bidirectional() { 2 } else { 1 };
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
    rnn_fwd_shape_check(x.shape(), hx.map(|hx| hx.shape()), &config);
    let rnn_exe = config.create_executor(is_training, x.shape()[1]);
    let reserve_size = rnn_exe.get_reserve_size();
    let workspace_size = rnn_exe.get_workspace_size();

    let reserve = Nvidia::alloc(reserve_size).unwrap();
    let workspace = Nvidia::alloc(workspace_size).unwrap();

    let mut y = Matrix::alloc([x.shape()[0], x.shape()[1], config.get_hidden_size()]);
    let mut hy = match hx {
        Some(hx) => Matrix::alloc([hx.shape()[0], hx.shape()[1]]),
        None => {
            let d = config.get_num_layers() * if config.is_bidirectional() { 2 } else { 1 };
            Matrix::alloc([d, config.get_hidden_size()])
        }
    };

    let output = rnn_exe.fwd(
        x.as_ptr(),
        y.to_ref_mut().as_mut_ptr(),
        hx.as_ptr(),
        hy.to_ref_mut().as_mut_ptr(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        weight,
        workspace,
        reserve,
    );
    RNNOutput { y, hy }
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
    rnn_fwd_shape_check(x.shape(), hx.map(|hx| hx.shape()), &config);
    let rnn_exe = config.create_executor(true, x.shape()[1]);
    let reserve_size = rnn_exe.get_reserve_size();
    let workspace_size = rnn_exe.get_workspace_size();

    let reserve = Nvidia::alloc(reserve_size).unwrap();
    let workspace = Nvidia::alloc(workspace_size).unwrap();

    let mut dx = Matrix::alloc(x.shape());
    let mut dhx = match hx {
        Some(hx) => Matrix::alloc(hx.shape()),
        None => {
            let d = config.get_num_layers() * if config.is_bidirectional() { 2 } else { 1 };
            Matrix::alloc([d, config.get_hidden_size()])
        }
    };

    let output = rnn_exe.bkwd_data(
        y.as_ptr(),
        dy.as_ptr(),
        dx.to_ref_mut().as_mut_ptr(),
        hx.as_ptr(),
        dhy.as_ptr(),
        dhx.to_ref_mut().as_mut_ptr(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        weight,
        workspace,
        reserve,
    );
    RNNBkwdDataOutput { dx, dhx }
}

pub fn rnn_bkwd_weights<T: Num>(
    x: Matrix<Ref<&T>, DimDyn, Nvidia>,
    hx: Option<Matrix<Ref<&T>, DimDyn, Nvidia>>,
    y: Matrix<Ref<&T>, DimDyn, Nvidia>,
    config: RNNConfig<T>,
) -> RNNParameters<T> {
    rnn_fwd_shape_check(x.shape(), hx.map(|hx| hx.shape()), &config);
    let rnn_exe = config.create_executor(true, x.shape()[1]);
    let reserve_size = rnn_exe.get_reserve_size();
    let workspace_size = rnn_exe.get_workspace_size();

    let reserve = Nvidia::alloc(reserve_size).unwrap();
    let workspace = Nvidia::alloc(workspace_size).unwrap();

    let mut dweight = Matrix::alloc(params.get_weight_shape());
    let mut dhx = match hx {
        Some(hx) => Matrix::alloc(hx.shape()),
        None => {
            let d = config.get_num_layers() * if config.is_bidirectional() { 2 } else { 1 };
            Matrix::alloc([d, config.get_hidden_size()])
        }
    };

    let output = rnn_exe.bkwd_weights(
        x.as_ptr(),
        hx.as_ptr(),
        y.as_ptr(),
        dweight.to_ref_mut().as_mut_ptr(),
        workspace.as_ptr(),
        reserve.as_ptr(),
    );
    RNNParameters {
        weight: dweight.as_mut_ptr(),
    }
}
