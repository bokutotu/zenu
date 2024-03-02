pub trait SoftMax {
    fn softmax(&mut self, source: Matrix<ViewMem<T>, DimDyn>);
    fn softmax_backward(&mut self, source: Matrix<ViewMem<T>, DimDyn>);
}
