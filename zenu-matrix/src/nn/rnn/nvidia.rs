use crate::{
    device::{nvidia::Nvidia, DeviceBase},
    dim::DimDyn,
    matrix::{Matrix, Ref},
    num::Num,
};

use super::{
    RNNBackwardWeightsOutputNvidia, RNNBkwdDataOutput, RNNBkwdWeightsOutput, RNNConfig, RNNOutput,
    RNNParameters, RNN,
};

impl RNN for Nvidia {
    fn fwd<T: Num>(
        x: Matrix<Ref<&T>, DimDyn, Self>,
        hx: Matrix<Ref<&T>, DimDyn, Self>,
        is_training: bool,
        params: RNNParameters<T, Self>,
        config: RNNConfig<T>,
    ) -> RNNOutput<T, Self> {
        match params {
            RNNParameters::Nvidia(params) => {
                let exe = config.config.create_executor(is_training, x.shape()[1]);
                let mut hy = Matrix::alloc([
                    config.config.config.num_layers,
                    config.config.config.hidden_size,
                ]);
                let mut y =
                    Matrix::alloc([x.shape()[0], x.shape()[1], config.config.config.hidden_size]);
                let workspace_size = exe.workspace.workspace_size;
                let reserve_size = exe.workspace.reserve_size;

                let workspace = <Self as DeviceBase>::alloc(workspace_size).unwrap() as *mut _;
                let reserve = <Self as DeviceBase>::alloc(reserve_size).unwrap() as *mut _;
                exe.fwd(
                    x.as_ptr(),
                    y.to_ref_mut().as_mut_ptr(),
                    hx.as_ptr(),
                    hy.to_ref_mut().as_mut_ptr(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    params.weight as *mut _,
                    workspace,
                    reserve,
                );

                RNNOutput { y, hy }
            }
            RNNParameters::Cpu(_) => panic!("RNN Parameters must be Nvidia"),
        }
    }

    fn bkwd_data<T: crate::num::Num>(
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&T>, DimDyn, Self>,
        dy: Matrix<Ref<&T>, DimDyn, Self>,
        hx: Matrix<Ref<&T>, DimDyn, Self>,
        dhy: Matrix<Ref<&T>, DimDyn, Self>,
        params: RNNParameters<T, Self>,
        config: RNNConfig<T>,
    ) -> RNNBkwdDataOutput<T, Self> {
        match params {
            RNNParameters::Nvidia(params) => {
                let exe = config.config.create_executor(true, x.shape()[1]);
                let mut dx = Matrix::alloc(x.shape());
                let mut dhx = Matrix::alloc(hx.shape());
                let workspace_size = exe.workspace.workspace_size;
                let reserve_size = exe.workspace.reserve_size;

                let workspace = <Self as DeviceBase>::alloc(workspace_size).unwrap() as *mut _;
                let reserve = <Self as DeviceBase>::alloc(reserve_size).unwrap() as *mut _;
                exe.bkwd_data(
                    y.as_ptr(),
                    dy.as_ptr(),
                    dx.to_ref_mut().as_mut_ptr(),
                    hx.as_ptr(),
                    dhy.as_ptr(),
                    dhx.to_ref_mut().as_mut_ptr(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    params.weight as *mut _,
                    workspace,
                    reserve,
                );

                RNNBkwdDataOutput { dx, dhx }
            }
            RNNParameters::Cpu(_) => panic!("RNN Parameters must be Nvidia"),
        }
    }

    fn bkwd_weights<T: crate::num::Num>(
        x: Matrix<Ref<&T>, DimDyn, Self>,
        hx: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&T>, DimDyn, Self>,
        params: RNNParameters<T, Self>,
        config: RNNConfig<T>,
    ) -> RNNBkwdWeightsOutput<T, Self> {
        match params {
            RNNParameters::Nvidia(_) => {
                let exe = config.config.create_executor(true, x.shape()[1]);
                let workspace_size = exe.workspace.workspace_size;
                let reserve_size = exe.workspace.reserve_size;

                let dweight_size = config.config.get_weight_bytes();
                let dwx = <Self as DeviceBase>::alloc(dweight_size).unwrap() as *mut _;

                let workspace = <Self as DeviceBase>::alloc(workspace_size).unwrap() as *mut _;
                let reserve = <Self as DeviceBase>::alloc(reserve_size).unwrap() as *mut _;
                exe.bkwd_weights(x.as_ptr(), hx.as_ptr(), y.as_ptr(), dwx, workspace, reserve);

                RNNBkwdWeightsOutput::Nvidia(RNNBackwardWeightsOutputNvidia { dw: dwx as *mut _ })
            }
            RNNParameters::Cpu(_) => panic!("RNN Parameters must be Nvidia"),
        }
    }
}
