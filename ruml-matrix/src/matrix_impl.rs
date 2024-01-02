use crate::{
    cpu_memory::{CpuOwnedMemory, CpuViewMemory},
    dim::{cal_offset, default_stride, DimTrait, LessDimTrait},
    dim_impl::{Dim1, Dim2, Dim3, Dim4},
    index::{IndexAxisTrait, ShapeStride, SliceTrait},
    matrix::{
        AsMutPtr, AsPtr, IndexAxis, IndexAxisMut, IndexItem, IndexItemAsign, MatrixBase,
        MatrixSlice, MatrixSliceMut, OwnedMatrix, ToOwnedMatrix, ToViewMatrix, ToViewMutMatrix,
        ViewMatrix, ViewMutMatix,
    },
    memory::{
        Memory, OwnedMemory, ToOwnedMemory, ToViewMemory, ToViewMutMemory, ViewMemory,
        ViewMutMemory,
    },
    num::Num,
};

pub struct Matrix<M, S> {
    memory: M,
    shape: S,
    stride: S,
}

impl<M, S> Matrix<M, S> {
    pub fn new(memory: M, shape: S, stride: S) -> Self {
        Matrix {
            memory,
            shape,
            stride,
        }
    }

    pub(crate) fn update_shape(&mut self, shape: S) {
        self.shape = shape;
    }

    pub(crate) fn update_stride(&mut self, stride: S) {
        self.stride = stride;
    }
}

impl<T: Num, M: Memory<Item = T>, S: DimTrait> MatrixBase for Matrix<M, S> {
    type Dim = S;
    type Item = T;

    fn shape_stride(&self) -> ShapeStride<Self::Dim> {
        ShapeStride::new(self.shape, self.stride)
    }

    fn is_default_stride(&self) -> bool {
        default_stride(self.shape_stride().shape()) == self.shape_stride().stride()
    }
}

impl<M: ToViewMemory, S: DimTrait> ToViewMatrix for Matrix<M, S> {
    type View<'a> = Matrix<M::View<'a>, S>
    where
        Self: 'a;

    fn to_view(&self) -> Self::View<'_> {
        Matrix {
            memory: self.memory.to_view(0),
            shape: self.shape,
            stride: self.stride,
        }
    }
}

impl<M: ToViewMutMemory, S: DimTrait> ToViewMutMatrix for Matrix<M, S> {
    type ViewMut<'a> = Matrix<M::ViewMut<'a>, S>
    where
        Self: 'a;

    fn to_view_mut(&mut self) -> Self::ViewMut<'_> {
        Matrix {
            memory: self.memory.to_view_mut(0),
            shape: self.shape,
            stride: self.stride,
        }
    }
}

impl<M: ToOwnedMemory, S: DimTrait> ToOwnedMatrix for Matrix<M, S> {
    type Owned = Matrix<M::Owned, S>;

    fn to_owned(&self) -> Self::Owned {
        Matrix {
            memory: self.memory.to_owned_memory(),
            shape: self.shape,
            stride: self.stride,
        }
    }
}

impl<M: ViewMutMemory, S: DimTrait> AsMutPtr for Matrix<M, S> {
    fn as_mut_ptr(&mut self) -> *mut Self::Item {
        self.memory.as_mut_ptr_offset(0)
    }
}

impl<M: Memory, S: DimTrait> AsPtr for Matrix<M, S> {
    fn as_ptr(&self) -> *const Self::Item {
        self.memory.as_ptr_offset(0)
    }
}

impl<M: ViewMemory, S: DimTrait> ViewMatrix for Matrix<M, S> {}

impl<M: ViewMutMemory, S: DimTrait> ViewMutMatix for Matrix<M, S> {}

impl<M: OwnedMemory, S: DimTrait> OwnedMatrix for Matrix<M, S> {
    fn from_vec(vec: Vec<Self::Item>, dim: Self::Dim) -> Self {
        if vec.len() != dim.num_elm() {
            panic!("vec.len() != dim.num_elm()");
        }
        let stride = default_stride(dim);
        let memory = M::from_vec(vec);
        Matrix {
            memory,
            shape: dim,
            stride,
        }
    }
}

impl<M: ToViewMemory, D: DimTrait, S: SliceTrait<Dim = D>> MatrixSlice<D, S> for Matrix<M, D> {
    type Output<'a> = Matrix<M::View<'a>, D>
    where
        Self: 'a;

    fn slice(&self, index: S) -> Self::Output<'_> {
        let shape_stride = self.shape_stride();
        let shape = shape_stride.shape();
        let stride = shape_stride.stride();
        let new_shape_stride = index.sliced_shape_stride(shape, stride);
        let offset = index.sliced_offset(stride, self.memory.get_offset());
        Self::Output::new(
            self.memory.to_view(offset),
            new_shape_stride.shape(),
            new_shape_stride.stride(),
        )
    }
}

impl<M: ToViewMutMemory, D: DimTrait, S: SliceTrait<Dim = D>> MatrixSliceMut<D, S>
    for Matrix<M, D>
{
    type Output<'a> = Matrix<M::ViewMut<'a>, D>
    where
        Self: 'a;

    fn slice_mut(&mut self, index: S) -> Self::Output<'_> {
        let shape_stride = self.shape_stride();
        let shape = shape_stride.shape();
        let stride = shape_stride.stride();
        let new_shape_stride = index.sliced_shape_stride(shape, stride);
        let offset = index.sliced_offset(stride, self.memory.get_offset());
        Self::Output::new(
            self.memory.to_view_mut(offset),
            new_shape_stride.shape(),
            new_shape_stride.stride(),
        )
    }
}

impl<I: IndexAxisTrait, M: ToViewMemory, D: DimTrait + LessDimTrait> IndexAxis<I> for Matrix<M, D>
where
    <D as LessDimTrait>::LessDim: DimTrait,
{
    type Output<'a> = Matrix<M::View<'a>, <D as LessDimTrait>::LessDim>
    where
        Self: 'a;

    fn index_axis(&self, index: I) -> Self::Output<'_> {
        let shape_stride = self.shape_stride();
        let shape = shape_stride.shape();
        let stride = shape_stride.stride();
        let new_shape_stride = index.get_shape_stride(shape, stride);
        let offset = index.offset(stride);
        Self::Output::new(
            self.memory.to_view(offset),
            new_shape_stride.shape(),
            new_shape_stride.stride(),
        )
    }
}

impl<I: IndexAxisTrait, M: ToViewMutMemory, D: DimTrait + LessDimTrait> IndexAxisMut<I>
    for Matrix<M, D>
where
    <D as LessDimTrait>::LessDim: DimTrait,
{
    type Output<'a> = Matrix<M::ViewMut<'a>, <D as LessDimTrait>::LessDim>
    where
        Self: 'a;

    fn index_axis_mut(&mut self, index: I) -> Self::Output<'_> {
        let shape_stride = self.shape_stride();
        let shape = shape_stride.shape();
        let stride = shape_stride.stride();
        let new_shape_stride = index.get_shape_stride(shape, stride);
        let offset = index.offset(stride);
        Self::Output::new(
            self.memory.to_view_mut(offset),
            new_shape_stride.shape(),
            new_shape_stride.stride(),
        )
    }
}

impl<D: DimTrait, M: Memory> IndexItem<D> for Matrix<M, D> {
    fn index_item(&self, index: D) -> Self::Item {
        if self.shape_stride().shape().is_overflow(index) {
            panic!("index is overflow");
        }

        let offset = cal_offset(index, self.shape_stride().stride());
        self.memory.value_offset(offset)
    }
}

impl<'a, T: Num, D: DimTrait, VM: ViewMutMemory + Memory<Item = T>> IndexItemAsign<D>
    for Matrix<VM, D>
{
    fn index_item_asign(&mut self, index: Self::Dim, value: Self::Item) {
        if self.shape_stride().shape().is_overflow(index) {
            panic!("index is overflow");
        }

        let offset = cal_offset(index, self.shape_stride().stride());
        unsafe {
            *self.memory.as_mut_ptr_offset(offset) = value;
        }
    }
}

pub type CpuOwnedMatrix1D<T> = Matrix<CpuOwnedMemory<T>, Dim1>;
pub type CpuViewMatrix1D<'a, T> = Matrix<CpuViewMemory<'a, T>, Dim1>;
pub type CpuOwnedMatrix2D<T> = Matrix<CpuOwnedMemory<T>, Dim2>;
pub type CpuViewMatrix2D<'a, T> = Matrix<CpuViewMemory<'a, T>, Dim2>;
pub type CpuOwnedMatrix3D<T> = Matrix<CpuOwnedMemory<T>, Dim3>;
pub type CpuViewMatrix3D<'a, T> = Matrix<CpuViewMemory<'a, T>, Dim3>;
pub type CpuOwnedMatrix4D<T> = Matrix<CpuOwnedMemory<T>, Dim4>;
pub type CpuViewMatrix4D<'a, T> = Matrix<CpuViewMemory<'a, T>, Dim4>;

#[cfg(test)]
mod matrix_slice {
    use crate::dim;
    use crate::dim_impl::Dim1;
    use crate::slice;

    use super::*;

    #[test]
    fn index_item_1d() {
        let m =
            CpuOwnedMatrix1D::from_vec(vec![1f32, 2., 3., 4., 5., 6., 7., 8., 9.], Dim1::new([9]));
        let item = m.index_item(Dim1::new([1]));
        assert_eq!(item, 2.);
    }

    #[test]
    fn slice_1d() {
        let m =
            CpuOwnedMatrix1D::from_vec(vec![1f32, 2., 3., 4., 5., 6., 7., 8., 9.], Dim1::new([9]));
        let s = m.slice(slice!(1..4));
        assert_eq!(s.index_item(dim!(0)), 2.);
        assert_eq!(s.index_item(dim!(1)), 3.);
        assert_eq!(s.index_item(dim!(2)), 4.);
        assert_eq!(s.shape_stride().shape()[0], 3);
        assert_eq!(s.shape_stride().stride()[0], 1);
    }

    #[test]
    fn slice_2d() {
        let m = CpuOwnedMatrix2D::from_vec(
            vec![1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            Dim2::new([3, 4]),
        );
        let s = m.slice(slice!(1..3, 1..4));
        assert_eq!(s.index_item(dim!(0, 0)), 6.);
        assert_eq!(s.index_item(dim!(0, 1)), 7.);
        assert_eq!(s.index_item(dim!(0, 2)), 8.);
        assert_eq!(s.index_item(dim!(1, 0)), 10.);
        assert_eq!(s.index_item(dim!(1, 1)), 11.);
        assert_eq!(s.index_item(dim!(1, 2)), 12.);
        assert_eq!(s.shape_stride().shape()[0], 2);
        assert_eq!(s.shape_stride().shape()[1], 3);
        assert_eq!(s.shape_stride().stride()[0], 4);
        assert_eq!(s.shape_stride().stride()[1], 1);
    }
}
