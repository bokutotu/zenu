use std::any::TypeId;

use crate::{
    cpu_memory::{CpuOwnedMemory, CpuViewMemory, CpuViewMutMemory},
    dim::{
        cal_offset, default_stride, Dim0, Dim1, Dim2, Dim3, Dim4, DimDyn, DimTrait, LessDimTrait,
    },
    index::{IndexAxisTrait, SliceTrait},
    matrix::{
        AsMutPtr, AsPtr, BlasMatrix, IndexAxis, IndexAxisMut, IndexItem, IndexItemAsign,
        MatrixBase, MatrixSlice, MatrixSliceDyn, MatrixSliceMut, OwnedMatrix, ToOwnedMatrix,
        ToViewMatrix, ToViewMutMatrix, ViewMatrix, ViewMutMatix,
    },
    memory::{
        Memory, OwnedMemory, ToOwnedMemory, ToViewMemory, ToViewMutMemory, ViewMemory,
        ViewMutMemory,
    },
    num::Num,
    shape_stride::ShapeStride,
    slice::Slice,
};

#[derive(Clone)]
pub struct Matrix<M, S> {
    memory: M,
    shape: S,
    stride: S,
}

impl<T, M> Matrix<M, Dim0>
where
    T: Num,
    M: Memory<Item = T>,
{
    pub fn scalar(scalar: T) -> Self
    where
        M: OwnedMemory<Item = T>,
    {
        let memory = M::from_vec(vec![scalar]);
        Matrix {
            memory,
            shape: Dim0::default(),
            stride: Dim0::default(),
        }
    }

    pub fn get_value(&self) -> T {
        self.memory.value_offset(0)
    }

    pub fn set_value(&mut self, value: T)
    where
        M: ViewMutMemory<Item = T>,
    {
        unsafe {
            self.as_mut_ptr().write(value);
        }
    }
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

    pub fn into_dyn_dim(self) -> Matrix<M, DimDyn>
    where
        M: Memory,
        S: DimTrait,
    {
        let shape = self.shape();
        let stride = self.stride();

        let mut shape_new = DimDyn::default();
        let mut stride_new = DimDyn::default();

        shape_new.set_len(shape.len());
        stride_new.set_len(stride.len());

        for i in 0..shape.len() {
            shape_new[i] = shape[i];
            stride_new[i] = stride[i];
        }

        Matrix::new(self.memory, shape_new, stride_new)
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
    fn from_vec<I: Into<Self::Dim>>(vec: Vec<Self::Item>, dim: I) -> Self {
        let dim = dim.into();
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

impl<M: ToViewMemory, D: DimTrait, S: SliceTrait<Dim = D>> MatrixSlice<S> for Matrix<M, D> {
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

impl<M: ToViewMutMemory, D: DimTrait, S: SliceTrait<Dim = D>> MatrixSliceMut<S> for Matrix<M, D> {
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

impl<D: DimTrait, M: Memory> IndexItem for Matrix<M, D> {
    fn index_item<I: Into<D>>(&self, index: I) -> Self::Item {
        let index = index.into();
        if self.shape_stride().shape().is_overflow(index) {
            panic!("index is overflow");
        }

        let offset = cal_offset(index, self.stride());
        self.memory.value_offset(offset)
    }
}

impl<T: Num, D: DimTrait, VM: ViewMutMemory + Memory<Item = T>> IndexItemAsign for Matrix<VM, D> {
    fn index_item_asign<I: Into<Self::Dim>>(&mut self, index: I, value: Self::Item) {
        let index = index.into();
        if self.shape_stride().shape().is_overflow(index) {
            panic!("index is overflow");
        }

        let offset = cal_offset(index, self.shape_stride().stride());
        unsafe {
            *self.memory.as_mut_ptr_offset(offset) = value;
        }
    }
}

pub(crate) fn matrix_into_dim<M: Memory, Dout: DimTrait, Din: DimTrait>(
    m: Matrix<M, Din>,
) -> Matrix<M, Dout> {
    if TypeId::of::<Din>() == TypeId::of::<Dout>() {
        let shape = m.shape();
        let stride = m.stride();

        let mut shape_new = Dout::default();
        let mut stride_new = Dout::default();

        for i in 0..shape.len() {
            shape_new[i] = shape[i];
            stride_new[i] = stride[i];
        }

        Matrix::new(m.memory, shape_new, stride_new)
    } else {
        panic!("Dout != Din");
    }
}

impl<T, M, D> MatrixSliceDyn for Matrix<M, D>
where
    T: Num,
    M: Memory<Item = T> + ToViewMemory,
    D: DimTrait,
{
    type Output<'a> = Matrix<M::View<'a>, DimDyn>
    where
        Self: 'a;
    fn slice_dyn(&self, index: Slice) -> Self::Output<'_> {
        let shape_stride = self.shape_stride().into_dyn();
        let new_shape_stride =
            index.sliced_shape_stride(shape_stride.shape(), shape_stride.stride());
        let offset = index.sliced_offset(shape_stride.stride(), self.memory.get_offset());
        Matrix::new(
            self.memory.to_view(offset),
            new_shape_stride.shape(),
            new_shape_stride.stride(),
        )
    }
}

impl<T: Num, M: Memory<Item = T>, D: DimTrait> BlasMatrix for Matrix<M, D> {
    type Blas = M::Blas;
}

pub type CpuOwnedMatrix0D<T> = Matrix<CpuOwnedMemory<T>, Dim0>;
pub type CpuViewMatrix0D<'a, T> = Matrix<CpuViewMemory<'a, T>, Dim0>;
pub type CpuViewMutMatrix0D<'a, T> = Matrix<CpuViewMutMemory<'a, T>, Dim0>;

pub type CpuOwnedMatrix1D<T> = Matrix<CpuOwnedMemory<T>, Dim1>;
pub type CpuViewMatrix1D<'a, T> = Matrix<CpuViewMemory<'a, T>, Dim1>;
pub type CpuViewMutMatrix1D<'a, T> = Matrix<CpuViewMutMemory<'a, T>, Dim1>;

pub type CpuOwnedMatrix2D<T> = Matrix<CpuOwnedMemory<T>, Dim2>;
pub type CpuViewMatrix2D<'a, T> = Matrix<CpuViewMemory<'a, T>, Dim2>;
pub type CpuViewMutMatrix2D<'a, T> = Matrix<CpuViewMutMemory<'a, T>, Dim2>;

pub type CpuOwnedMatrix3D<T> = Matrix<CpuOwnedMemory<T>, Dim3>;
pub type CpuViewMatrix3D<'a, T> = Matrix<CpuViewMemory<'a, T>, Dim3>;
pub type CpuViewMutMatrix3D<'a, T> = Matrix<CpuViewMutMemory<'a, T>, Dim3>;

pub type CpuOwnedMatrix4D<T> = Matrix<CpuOwnedMemory<T>, Dim4>;
pub type CpuViewMatrix4D<'a, T> = Matrix<CpuViewMemory<'a, T>, Dim4>;
pub type CpuViewMutMatrix4D<'a, T> = Matrix<CpuViewMutMemory<'a, T>, Dim4>;

pub type CpuOwnedMatrixDyn<T> = Matrix<CpuOwnedMemory<T>, DimDyn>;
pub type CpuViewMatrixDyn<'a, T> = Matrix<CpuViewMemory<'a, T>, DimDyn>;
pub type CpuViewMutMatrixDyn<'a, T> = Matrix<CpuViewMutMemory<'a, T>, DimDyn>;

#[cfg(test)]
mod matrix_slice {
    use crate::dim::Dim1;
    use crate::slice;
    use crate::slice_dynamic;

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
        assert_eq!(s.index_item([0]), 2.);
        assert_eq!(s.index_item([1]), 3.);
        assert_eq!(s.index_item([2]), 4.);
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
        assert_eq!(s.index_item([0, 0]), 6.);
        assert_eq!(s.index_item([0, 1]), 7.);
        assert_eq!(s.index_item([0, 2]), 8.);
        assert_eq!(s.index_item([1, 0]), 10.);
        assert_eq!(s.index_item([1, 1]), 11.);
        assert_eq!(s.index_item([1, 2]), 12.);
        assert_eq!(s.shape_stride().shape()[0], 2);
        assert_eq!(s.shape_stride().shape()[1], 3);
        assert_eq!(s.shape_stride().stride()[0], 4);
        assert_eq!(s.shape_stride().stride()[1], 1);
    }

    #[test]
    fn slice_dyn() {
        // define 4d matrix
        let m: CpuOwnedMatrix4D<f32> = CpuOwnedMatrix4D::from_vec(
            vec![
                1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34.,
                35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50.,
                51., 52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64.,
            ],
            [2, 2, 4, 4],
        );

        // into dyn
        let m = m.into_dyn_dim();
        let s = m.slice_dyn(slice_dynamic!(.., .., 2, ..));

        assert_eq!(s.index_item([0, 0, 0]), 9.);
        assert_eq!(s.index_item([0, 0, 1]), 10.);
        assert_eq!(s.index_item([0, 0, 2]), 11.);
        assert_eq!(s.index_item([0, 0, 3]), 12.);
        assert_eq!(s.index_item([0, 1, 0]), 25.);
        assert_eq!(s.index_item([0, 1, 1]), 26.);
        assert_eq!(s.index_item([0, 1, 2]), 27.);
        assert_eq!(s.index_item([0, 1, 3]), 28.);
        assert_eq!(s.index_item([1, 0, 0]), 41.);
        assert_eq!(s.index_item([1, 0, 1]), 42.);
        assert_eq!(s.index_item([1, 0, 2]), 43.);
        assert_eq!(s.index_item([1, 0, 3]), 44.);
        assert_eq!(s.index_item([1, 1, 0]), 57.);
        assert_eq!(s.index_item([1, 1, 1]), 58.);
        assert_eq!(s.index_item([1, 1, 2]), 59.);
        assert_eq!(s.index_item([1, 1, 3]), 60.);
    }
}
