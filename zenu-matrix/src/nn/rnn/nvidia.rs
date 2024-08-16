use crate::{
    device::nvidia::Nvidia,
    dim::DimDyn,
    matrix::{Matrix, Ref},
    num::Num,
};

use super::{RNNBkwdDataOutput, RNNBkwdWeightsOutput, RNNConfig, RNNOutput, RNNParameters, RNN};

impl RNN for Nvidia {
    fn fwd<T: Num>(
        &self,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        hx: Matrix<Ref<&T>, DimDyn, Self>,
        params: RNNParameters<T, Self>,
        config: RNNConfig<T>,
    ) -> RNNOutput<T, Self> {
        todo!()
    }

    fn bkwd_data<T: crate::num::Num>(
        &self,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&T>, DimDyn, Self>,
        hx: Matrix<Ref<&T>, DimDyn, Self>,
        dhy: Matrix<Ref<&T>, DimDyn, Self>,
        params: RNNParameters<T, Self>,
        config: RNNConfig<T>,
    ) -> RNNBkwdDataOutput<T, Self> {
        todo!()
    }

    fn bkwd_weights<T: crate::num::Num>(
        &self,
        x: Matrix<Ref<&T>, DimDyn, Self>,
        hx: Matrix<Ref<&T>, DimDyn, Self>,
        y: Matrix<Ref<&T>, DimDyn, Self>,
        params: RNNParameters<T, Self>,
        config: RNNConfig<T>,
    ) -> RNNBkwdWeightsOutput<T, Self> {
        todo!()
    }
}
