use crate::{dim::DimDyn, num::Num};

use super::{params::Params, Descriptor};

impl<T: Num, P: Params> Descriptor<T, P> {
    fn lstm_fwd_shape_check(&self, x: DimDyn, hx: Option<DimDyn>, cx: Option<DimDyn>) {
        todo!();
    }

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
        todo!();
    }

    fn lstm_bkwd_weights_shape_check(
        &self,
        x: DimDyn,
        hx: Option<DimDyn>,
        cx: Option<DimDyn>,
        y: DimDyn,
    ) {
        todo!();
    }
}
