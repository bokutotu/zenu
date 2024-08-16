use crate::{
    device::DeviceBase,
    dim::DimDyn,
    matrix::{Matrix, Ref},
    num::Num,
};

use super::{RNNBkwdDataOutput, RNNBkwdWeightsOutput, RNNConfig, RNNOutput, RNNParameters};

pub trait RNN: DeviceBase {
    fn fwd<T: Num>(
        &self,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        hx: Matrix<Ref<&T>, DimDyn, Self>,
        params: RNNParameters<T, Self>,
        config: RNNConfig<T>,
    ) -> RNNOutput<T, Self>;

    fn bkwd_data<T: Num>(
        &self,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&T>, DimDyn, Self>,
        hx: Matrix<Ref<&T>, DimDyn, Self>,
        dhy: Matrix<Ref<&T>, DimDyn, Self>,
        params: RNNParameters<T, Self>,
        config: RNNConfig<T>,
    ) -> RNNBkwdDataOutput<T, Self>;

    fn bkwd_weights<T: Num>(
        &self,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        hx: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&T>, DimDyn, Self>,
        params: RNNParameters<T, Self>,
        config: RNNConfig<T>,
    ) -> RNNBkwdWeightsOutput<T, Self>;
}
