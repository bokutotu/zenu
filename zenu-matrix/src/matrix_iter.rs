use crate::{
    dim::{DimDyn, DimTrait},
    matrix::MatrixBase,
    matrix_impl::Matrix,
    memory::{Memory, ToViewMemory, ToViewMutMemory},
};

pub struct MatrixIterByAxis<'a, M: Memory, D: DimTrait> {
    matrix: &'a Matrix<M, D>,
    axis: usize,
    len: usize,
    idx: usize,
}

impl<'a, M: Memory, D: DimTrait> MatrixIterByAxis<'a, M, D> {
    fn new(matrix: &'a Matrix<M, D>, axis: usize, len: usize) -> Self {
        Self {
            matrix,
            axis,
            len,
            idx: 0,
        }
    }
}

pub struct MatrixInterByAxisMut<'a, M: Memory, D: DimTrait> {
    matrix: &'a mut Matrix<M, D>,
    axis: usize,
    len: usize,
    idx: usize,
}

impl<'a, M: Memory, D: DimTrait> MatrixInterByAxisMut<'a, M, D> {
    fn new(matrix: &'a mut Matrix<M, D>, axis: usize, len: usize) -> Self {
        Self {
            matrix,
            axis,
            len,
            idx: 0,
        }
    }
}

impl<'a, M: ToViewMemory, D: DimTrait> Iterator for MatrixIterByAxis<'a, M, D> {
    type Item = Matrix<M::View<'a>, DimDyn>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.len {
            let stride = self.matrix.stride()[self.axis];
            let offset = self.idx * stride;
            let view = self.matrix.memory().to_view(offset);
            self.idx += 1;
            let new_shape = DimDyn::from(&[self.matrix.shape()[self.axis]]);
            let new_stride = DimDyn::from(&[stride]);
            Some(Matrix::new(view, new_shape, new_stride))
        } else {
            None
        }
    }
}

// impl<'a, M: ToViewMutMemory, D: DimTrait> Iterator for MatrixInterByAxisMut<'a, M, D> {
//     type Item = Matrix<M::ViewMut<'a>, DimDyn>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.idx < self.len {
//             let stride = self.matrix.stride()[self.axis];
//             let offset = self.idx * stride;
//             let shape = DimDyn::from(&[self.matrix.shape()[self.axis]]);
//             let view = self.matrix.memory_mut().to_view_mut(offset);
//             self.idx += 1;
//             let new_stride = DimDyn::from(&[stride]);
//             Some(Matrix::new(view, shape, new_stride))
//         } else {
//             None
//         }
//     }
// }
