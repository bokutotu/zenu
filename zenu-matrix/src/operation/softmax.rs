use crate::{dim::DimDyn, matrix_impl::Matrix, memory_impl::ViewMem, num::Num};

pub trait SoftMax<T: Num> {
    fn softmax(&mut self, source: Matrix<ViewMem<T>, DimDyn>);
    fn softmax_backward(&mut self, source: Matrix<ViewMem<T>, DimDyn>);
}
