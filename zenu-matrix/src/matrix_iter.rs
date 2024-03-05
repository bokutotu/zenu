use crate::{
    dim::{DimDyn, DimTrait},
    matrix::{MatrixBase, ToViewMutMatrix},
    matrix_impl::Matrix,
    memory::ToViewMutMemory,
    memory_impl::ViewMutMem,
    num::Num,
    shape_stride::ShapeStride,
};

struct MapAxis<'a, T: Num, F>
where
    F: FnMut(Matrix<ViewMutMem<T>, DimDyn>),
{
    matrix: Matrix<ViewMutMem<'a, T>, DimDyn>,
    axis: usize,
    fn_map: F,
}

impl<'a, T: Num, F> MapAxis<'a, T, F>
where
    F: FnMut(Matrix<ViewMutMem<T>, DimDyn>),
{
    fn new(matrix: Matrix<ViewMutMem<'a, T>, DimDyn>, axis: usize, fn_map: F) -> Self {
        Self {
            matrix,
            axis,
            fn_map,
        }
    }

    fn target_shape_stride(&self) -> ShapeStride<DimDyn> {
        let sh = self.target_shape();
        let st = self.target_stride();
        ShapeStride::new(DimDyn::from([sh]), DimDyn::from([st]))
    }

    fn target_stride(&self) -> usize {
        self.matrix.stride()[self.axis]
    }

    fn target_shape(&self) -> usize {
        self.matrix.shape()[self.axis]
    }

    fn target_offset(&self, index: usize) -> usize {
        self.target_shape() * self.target_stride() * index
    }

    fn num_loop(&self) -> usize {
        self.matrix.shape().num_elm() / self.target_shape()
    }

    fn apply(&mut self) {
        let shapt_stride = self.target_shape_stride();
        for idx in 0..self.num_loop() {
            let offset = self.target_offset(idx);
            let m = self.matrix.memory_mut();
            let view = m.to_view_mut(offset);
            let matrix = Matrix::new(view, shapt_stride.shape(), shapt_stride.stride());
            (self.fn_map)(matrix);
        }
    }
}

pub trait MatrixIter<T: Num> {
    fn map_axis<F>(&mut self, axis: usize, fn_map: F)
    where
        F: FnMut(Matrix<ViewMutMem<T>, DimDyn>);
}

impl<T: Num, M: ToViewMutMemory<Item = T>> MatrixIter<T> for Matrix<M, DimDyn> {
    fn map_axis<F>(&mut self, axis: usize, fn_map: F)
    where
        F: FnMut(Matrix<ViewMutMem<T>, DimDyn>),
    {
        let mut_matrix = self.to_view_mut();
        let mut map_axis = MapAxis::new(mut_matrix, axis, fn_map);
        map_axis.apply();
    }
}
