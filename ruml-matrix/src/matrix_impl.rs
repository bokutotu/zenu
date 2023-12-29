use crate::{
    cpu_memory::{CpuOwnedMemory, CpuViewMemory},
    dim::{cal_offset, default_stride, DimTrait, LessDimTrait},
    dim_impl::{Dim1, Dim2, Dim3, Dim4},
    index::{IndexAxisTrait, ShapeStride, SliceTrait},
    matrix::{
        IndexAxis, IndexAxisMut, IndexItem, MatrixBase, MatrixSlice, MatrixSliceMut, OwnedMatrix,
        ViewMatrix, ViewMutMatix,
    },
    memory::{Memory, OwnedMemory, ToViewMemory, ToViewMutMemory, ViewMemory, ViewMutMemory},
};

pub struct Matrix<M, S> {
    memory: M,
    shape: S,
    stride: S,
}

pub type CpuOwnedMatrix1D<T> = Matrix<CpuOwnedMemory<T>, Dim1>;
pub type CpuViewMatrix1D<'a, T> = Matrix<CpuViewMemory<'a, T>, Dim1>;
pub type CpuOwnedMatrix2D<T> = Matrix<CpuOwnedMemory<T>, Dim2>;
pub type CpuViewMatrix2D<'a, T> = Matrix<CpuViewMemory<'a, T>, Dim2>;
pub type CpuOwnedMatrix3D<T> = Matrix<CpuOwnedMemory<T>, Dim3>;
pub type CpuViewMatrix3D<'a, T> = Matrix<CpuViewMemory<'a, T>, Dim3>;
pub type CpuOwnedMatrix4D<T> = Matrix<CpuOwnedMemory<T>, Dim4>;
pub type CpuViewMatrix4D<'a, T> = Matrix<CpuViewMemory<'a, T>, Dim4>;

impl<M, S> MatrixBase for Matrix<M, S>
where
    M: Memory,
    S: DimTrait,
{
    type Memory = M;
    type Dim = S;
    fn memory(&self) -> &Self::Memory {
        &self.memory
    }

    fn shape_stride(&self) -> ShapeStride<Self::Dim> {
        ShapeStride::new(self.shape, self.stride)
    }

    fn construct(memory: Self::Memory, shape: Self::Dim, stride: Self::Dim) -> Self {
        Matrix {
            memory,
            shape,
            stride,
        }
    }

    fn memory_mut(&mut self) -> &mut Self::Memory {
        &mut self.memory
    }
}

impl<M, S> OwnedMatrix for Matrix<M, S>
where
    M: OwnedMemory + ToViewMemory + ToViewMutMemory + Clone,
    S: DimTrait,
{
    type View<'a> = Matrix<<M as ToViewMemory>::View<'a>,S>
    where
        Self: 'a;

    type ViewMut<'a> = Matrix<<M as ToViewMutMemory>::ViewMut<'a>, S>
    where
        Self: 'a;

    fn to_view(&self) -> Self::View<'_> {
        Matrix {
            memory: self.memory.to_view(0),
            shape: self.shape,
            stride: self.stride,
        }
    }
    fn to_view_mut(&mut self) -> Self::ViewMut<'_> {
        Matrix {
            memory: self.memory.to_view_mut(0),
            shape: self.shape,
            stride: self.stride,
        }
    }
    fn from_vec(vec: Vec<<<Self as MatrixBase>::Memory as Memory>::Item>, dim: Self::Dim) -> Self {
        let memory = M::from_vec(vec);
        let stride = default_stride(dim);
        Matrix {
            memory,
            shape: dim,
            stride,
        }
    }
}

impl<M, S> ViewMatrix for Matrix<M, S>
where
    M: ViewMemory,
    S: DimTrait,
{
    type Owned = Matrix<M::Owned, S>;

    fn to_owned(&self) -> Self::Owned {
        Matrix {
            memory: self.memory.to_owned_memory(),
            shape: self.shape,
            stride: self.stride,
        }
    }
}

impl<M, S> ViewMutMatix for Matrix<M, S>
where
    M: ViewMutMemory,
    S: DimTrait,
{
    fn view_mut_memory(&self) -> &Self::Memory {
        &self.memory
    }
}

impl<M, S, SS> MatrixSlice<M, S, SS> for Matrix<M, S>
where
    M: ToViewMemory,
    S: DimTrait,
    SS: SliceTrait<Dim = S>,
{
    type Output<'a> = Matrix<<M as ToViewMemory>::View<'a>, S>
    where
        Self: 'a;

    fn slice(&self, index: SS) -> Self::Output<'_> {
        let shape_stride = self.shape_stride();
        let shape = shape_stride.shape();
        let stride = shape_stride.stride();
        let new_shape_stride = index.sliced_shape_stride(shape, stride);
        let offset = index.sliced_offset(stride, self.memory().get_offset());
        Self::Output::construct(
            self.memory().to_view(offset),
            new_shape_stride.shape(),
            new_shape_stride.stride(),
        )
    }
}

impl<M, S, SS> MatrixSliceMut<S, SS> for Matrix<M, S>
where
    M: ToViewMutMemory,
    S: DimTrait,
    SS: SliceTrait<Dim = S>,
{
    type Output<'a> = Matrix<
        <<Matrix<M, S> as MatrixBase>::Memory as ToViewMutMemory>::ViewMut<'a>,
        S
    >
    where
        Self: 'a;

    fn slice_mut(&mut self, index: SS) -> Self::Output<'_> {
        let shape_stride = self.shape_stride();
        let shape = shape_stride.shape();
        let stride = shape_stride.stride();
        let new_shape_stride = index.sliced_shape_stride(shape, stride);
        let offset = index.sliced_offset(stride, self.memory().get_offset());
        Self::Output::construct(
            self.memory_mut().to_view_mut(offset),
            new_shape_stride.shape(),
            new_shape_stride.stride(),
        )
    }
}

impl<M, S, I> IndexAxis<I> for Matrix<M, S>
where
    M: ToViewMemory,
    S: DimTrait + LessDimTrait,
    I: IndexAxisTrait,
{
    type Output<'a> = Matrix<
        <<Matrix<M, S> as MatrixBase>::Memory as ToViewMemory>::View<'a>,
        <<Matrix<M, S> as MatrixBase>::Dim as LessDimTrait>::LessDim,
    >
    where
        Self: 'a;

    fn index_axis(&self, index: I) -> Self::Output<'_> {
        let shape_stride = self.shape_stride();
        let shape = shape_stride.shape();
        let stride = shape_stride.stride();
        let new_shape_stride = index.get_shape_stride(shape, stride);
        let offset = index.offset(stride);
        Self::Output::construct(
            self.memory().to_view(offset),
            new_shape_stride.shape(),
            new_shape_stride.stride(),
        )
    }
}

impl<M, S, I> IndexAxisMut<I> for Matrix<M, S>
where
    M: ToViewMutMemory,
    S: DimTrait + LessDimTrait,
    I: IndexAxisTrait,
{
    type Output<'a> = Matrix<
        <<Matrix<M, S> as MatrixBase>::Memory as ToViewMutMemory>::ViewMut<'a>,
        <<Matrix<M, S> as MatrixBase>::Dim as LessDimTrait>::LessDim,
    >
    where
        Self: 'a;

    fn index_axis_mut(&mut self, index: I) -> Self::Output<'_> {
        let shape_stride = self.shape_stride();
        let shape = shape_stride.shape();
        let stride = shape_stride.stride();
        let new_shape_stride = index.get_shape_stride(shape, stride);
        let offset = index.offset(stride);
        Self::Output::construct(
            self.memory_mut().to_view_mut(offset),
            new_shape_stride.shape(),
            new_shape_stride.stride(),
        )
    }
}

impl<M, D> IndexItem<D> for Matrix<M, D>
where
    D: DimTrait,
    M: Memory,
{
    fn index_item(&self, index: D) -> <Self::Memory as Memory>::Item {
        if self.shape_stride().shape().is_overflow(index) {
            panic!("index out of bounds");
        }

        let offset = cal_offset(index, self.shape_stride().stride());
        self.memory().ptr_offset(offset)
    }
}

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
