use std::{any::TypeId, marker::PhantomData};

use crate::{
    device::{Device, DeviceBase},
    dim::{cal_offset, default_stride, DimDyn, DimTrait, LessDimTrait},
    index::{IndexAxisTrait, SliceTrait},
    num::Num,
    shape_stride::ShapeStride,
    slice::Slice,
};

#[cfg(feature = "nvidia")]
use crate::device::nvidia::Nvidia;

pub trait Repr: Default {
    type Item: Num;

    fn drop_memory<D: DeviceBase>(ptr: *mut Self::Item, _: D);
    fn clone_memory<D: DeviceBase>(ptr: *mut Self::Item, len: usize, _: D) -> *mut Self::Item;
}

pub trait OwnedRepr: Repr {}

pub struct Owned<T: Num> {
    _maker: PhantomData<T>,
}

pub struct Ref<A> {
    _maker: PhantomData<A>,
}

impl<T: Num> Default for Owned<T> {
    fn default() -> Self {
        Owned {
            _maker: PhantomData,
        }
    }
}

impl<A> Default for Ref<A> {
    fn default() -> Self {
        Ref {
            _maker: PhantomData,
        }
    }
}

impl<'a, T: Num> Repr for Ref<&'a T> {
    type Item = T;

    fn drop_memory<D: DeviceBase>(_ptr: *mut Self::Item, _: D) {}
    fn clone_memory<D: DeviceBase>(ptr: *mut Self::Item, _len: usize, _: D) -> *mut Self::Item {
        ptr
    }
}

impl<'a, T: Num> Repr for Ref<&'a mut T> {
    type Item = T;

    fn drop_memory<D: DeviceBase>(_ptr: *mut Self::Item, _: D) {}
    fn clone_memory<D: DeviceBase>(ptr: *mut Self::Item, _len: usize, _: D) -> *mut Self::Item {
        ptr
    }
}

impl<T: Num> Repr for Owned<T> {
    type Item = T;

    fn drop_memory<D: DeviceBase>(ptr: *mut Self::Item, _: D) {
        D::drop_ptr(ptr);
    }

    fn clone_memory<D: DeviceBase>(ptr: *mut Self::Item, len: usize, _: D) -> *mut Self::Item {
        D::clone_ptr(ptr, len)
    }
}

impl<T: Num> OwnedRepr for Owned<T> {}

pub struct Ptr<R, D>
where
    R: Repr,
    D: DeviceBase,
{
    ptr: *mut R::Item,
    len: usize,
    offset: usize,
    repr: PhantomData<R>,
    device: PhantomData<D>,
}

impl<R, D> Ptr<R, D>
where
    R: Repr,
    D: DeviceBase,
{
    pub(crate) fn new(ptr: *mut R::Item, len: usize, offset: usize) -> Self {
        Ptr {
            ptr,
            len,
            offset,
            repr: PhantomData,
            device: PhantomData,
        }
    }

    #[must_use]
    pub fn offset_ptr(&self, offset: usize) -> Ptr<Ref<&R::Item>, D> {
        Ptr {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset + offset,
            repr: PhantomData,
            device: PhantomData,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[expect(clippy::missing_panics_doc)]
    #[must_use]
    pub fn get_item(&self, offset: usize) -> R::Item {
        assert!(offset < self.len, "Index out of bounds");
        D::get_item(self.ptr, offset + self.offset)
    }

    fn to_ref<'a>(&self) -> Ptr<Ref<&'a R::Item>, D> {
        Ptr {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset,
            repr: PhantomData,
            device: PhantomData,
        }
    }

    fn to<Dout: DeviceBase>(&self) -> Ptr<Owned<R::Item>, Dout> {
        #[cfg(feature = "nvidia")]
        use crate::device::cpu::Cpu;

        let self_raw_ptr = self.ptr;
        let len = self.len;

        let ptr = match (TypeId::of::<D>(), TypeId::of::<Dout>()) {
            (a, b) if a == b => Owned::clone_memory(self_raw_ptr, len, D::default()),
            #[cfg(feature = "nvidia")]
            (a, b) if a == TypeId::of::<Cpu>() && b == TypeId::of::<Nvidia>() => {
                zenu_cuda::runtime::copy_to_gpu(self_raw_ptr, len)
            }
            #[cfg(feature = "nvidia")]
            (a, b) if a == TypeId::of::<Nvidia>() && b == TypeId::of::<Cpu>() => {
                zenu_cuda::runtime::copy_to_cpu(self_raw_ptr, len)
            }
            _ => unreachable!(),
        };

        Ptr::new(ptr, len, self.offset)
    }
}

impl<R, D> Drop for Ptr<R, D>
where
    R: Repr,
    D: DeviceBase,
{
    fn drop(&mut self) {
        R::drop_memory(self.ptr, D::default());
    }
}

impl<'a, T: Num, D: DeviceBase> Ptr<Ref<&'a mut T>, D> {
    #[must_use]
    pub fn offset_ptr_mut(self, offset: usize) -> Ptr<Ref<&'a mut T>, D> {
        Ptr {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset + offset,
            repr: PhantomData,
            device: PhantomData,
        }
    }

    #[expect(clippy::missing_panics_doc)]
    pub fn assign_item(&self, offset: usize, value: T) {
        assert!(offset < self.len, "Index out of bounds");
        D::assign_item(self.ptr, offset + self.offset, value);
    }
}

impl<R, D> Clone for Ptr<R, D>
where
    R: Repr,
    D: DeviceBase,
{
    fn clone(&self) -> Self {
        Ptr {
            ptr: R::clone_memory(self.ptr, self.len, D::default()),
            len: self.len,
            offset: self.offset,
            repr: PhantomData,
            device: PhantomData,
        }
    }
}

impl<R, D> Ptr<R, D>
where
    R: OwnedRepr,
    D: DeviceBase,
{
    fn to_ref_mut<'a>(&mut self) -> Ptr<Ref<&'a mut R::Item>, D> {
        Ptr {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset,
            repr: PhantomData,
            device: PhantomData,
        }
    }
}

pub struct Matrix<R, S, D>
where
    R: Repr,
    S: DimTrait,
    D: DeviceBase,
{
    ptr: Ptr<R, D>,
    shape: S,
    stride: S,
}

impl<R, S, D> Clone for Matrix<R, S, D>
where
    R: Repr,
    S: DimTrait,
    D: DeviceBase,
{
    fn clone(&self) -> Self {
        Matrix {
            ptr: self.ptr.clone(),
            shape: self.shape,
            stride: self.stride,
        }
    }
}

impl<R, S, D> Matrix<R, S, D>
where
    R: Repr,
    S: DimTrait,
    D: DeviceBase,
{
    pub(crate) fn new(ptr: Ptr<R, D>, shape: S, stride: S) -> Self {
        Matrix { ptr, shape, stride }
    }

    pub(crate) unsafe fn ptr(&self) -> &Ptr<R, D> {
        &self.ptr
    }

    pub fn offset(&self) -> usize {
        self.ptr.offset
    }

    pub fn shape_stride(&self) -> ShapeStride<S> {
        ShapeStride::new(self.shape, self.stride)
    }

    pub fn shape(&self) -> S {
        self.shape
    }

    pub fn stride(&self) -> S {
        self.stride
    }

    pub fn is_default_stride(&self) -> bool {
        self.shape_stride().is_default_stride()
    }

    pub fn is_transpose_default_stride(&self) -> bool {
        self.shape_stride().is_transposed_default_stride()
    }

    pub fn as_ptr(&self) -> *const R::Item {
        unsafe { self.ptr.ptr.add(self.offset()) }
    }

    /// this code retunrs a slice of the matrix
    /// WARNING: even if the matrix has offset, the slice will be created from the original pointer
    pub fn to_vec(&self) -> Vec<R::Item>
    where
        R::Item: Clone,
    {
        let ptr_len = self.ptr.len();
        let mut vec = Vec::with_capacity(ptr_len);
        let non_offset_ptr = Ptr::<Ref<&R::Item>, D>::new(self.ptr.ptr, ptr_len, 0);
        for i in 0..ptr_len {
            vec.push(non_offset_ptr.get_item(i));
        }
        vec
    }

    pub fn into_dyn_dim(self) -> Matrix<R, DimDyn, D> {
        let mut shape = DimDyn::default();
        let mut stride = DimDyn::default();

        for i in 0..self.shape.len() {
            shape.push_dim(self.shape[i]);
            stride.push_dim(self.stride[i]);
        }
        Matrix {
            ptr: self.ptr,
            shape,
            stride,
        }
    }

    pub fn update_shape_stride(&mut self, shape_stride: ShapeStride<S>) {
        self.shape = shape_stride.shape();
        self.stride = shape_stride.stride();
    }

    pub fn update_shape(&mut self, shape: S) {
        self.shape = shape;
        self.stride = default_stride(shape);
    }

    pub fn update_stride(&mut self, stride: S) {
        self.stride = stride;
    }

    pub fn into_dim<S2>(self) -> Matrix<R, S2, D>
    where
        S2: DimTrait,
    {
        Matrix {
            ptr: self.ptr,
            shape: S2::from(self.shape.slice()),
            stride: S2::from(self.stride.slice()),
        }
    }

    pub fn slice<I>(&self, index: I) -> Matrix<Ref<&R::Item>, S, D>
    where
        I: SliceTrait<Dim = S>,
    {
        let shape = self.shape();
        let stride = self.stride();
        let new_shape_stride = index.sliced_shape_stride(shape, stride);
        let offset = index.sliced_offset(stride);
        Matrix {
            ptr: self.ptr.offset_ptr(offset),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
        }
    }

    pub fn slice_dyn(&self, index: Slice) -> Matrix<Ref<&R::Item>, DimDyn, D> {
        let shape_stride = self.shape_stride().into_dyn();
        let new_shape_stride =
            index.sliced_shape_stride(shape_stride.shape(), shape_stride.stride());
        let offset = index.sliced_offset(shape_stride.stride());
        Matrix {
            ptr: self.ptr.offset_ptr(offset),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
        }
    }

    pub fn index_axis<I>(&self, index: I) -> Matrix<Ref<&R::Item>, S, D>
    where
        I: IndexAxisTrait,
        S: LessDimTrait,
        S::LessDim: DimTrait,
    {
        let shape = self.shape();
        let stride = self.stride();
        let new_shape_stride = index.get_shape_stride(shape, stride);
        let offset = index.offset(stride);
        Matrix {
            ptr: self.ptr.offset_ptr(offset),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
        }
    }

    pub fn index_axis_dyn<I>(&self, index: I) -> Matrix<Ref<&R::Item>, DimDyn, D>
    where
        I: IndexAxisTrait,
    {
        let shape_stride = self.shape_stride().into_dyn();
        let new_shape_stride = index.get_shape_stride(shape_stride.shape(), shape_stride.stride());
        let offset = index.offset(shape_stride.stride());
        Matrix {
            ptr: self.ptr.offset_ptr(offset),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
        }
    }

    #[expect(clippy::missing_panics_doc)]
    pub fn index_item<I: Into<S>>(&self, index: I) -> R::Item {
        let index = index.into();
        assert!(!self.shape().is_overflow(index), "Index out of bounds");
        let offset = cal_offset(index, self.stride());
        self.ptr.get_item(offset)
    }

    pub fn to_ref<'a>(&self) -> Matrix<Ref<&'a R::Item>, S, D> {
        Matrix {
            ptr: self.ptr.to_ref(),
            shape: self.shape,
            stride: self.stride,
        }
    }

    pub fn convert_dim_type<Dout: DimTrait>(self) -> Matrix<R, Dout, D> {
        Matrix {
            ptr: self.ptr,
            shape: Dout::from(self.shape.slice()),
            stride: Dout::from(self.stride.slice()),
        }
    }

    pub fn new_matrix(&self) -> Matrix<Owned<R::Item>, S, D>
    where
        D: Device,
    {
        let mut owned = Matrix::zeros(self.shape());
        owned.to_ref_mut().copy_from(self);
        owned
    }

    #[expect(clippy::missing_errors_doc)]
    pub fn try_to_scalar(&self) -> Result<R::Item, String> {
        if self.shape().is_scalar() {
            let scalr = self.ptr.get_item(0);
            Ok(scalr)
        } else {
            Err("this matrix is not scalar".to_string())
        }
    }

    #[expect(clippy::missing_panics_doc)]
    pub fn to_scalar(&self) -> R::Item {
        if let Ok(scalar) = self.try_to_scalar() {
            scalar
        } else {
            panic!("Matrix is not scalar");
        }
    }

    #[expect(clippy::missing_panics_doc)]
    pub fn as_slice(&self) -> &[R::Item] {
        // let num_elm = self.shape().num_elm();
        // unsafe { std::slice::from_raw_parts(self.as_ptr(), num_elm) }
        if self.shape().len() <= 1 {
            // let num_elm = std::cmp::max(self.shape().num_elm(), 1);
            // unsafe { std::slice::from_raw_parts(self.as_ptr(), num_elm) }
            self.as_slice_unchecked()
        } else {
            panic!("Invalid shape");
        }
    }

    fn as_slice_unchecked(&self) -> &[R::Item] {
        let num_elm = self.shape().num_elm();
        unsafe { std::slice::from_raw_parts(self.as_ptr(), num_elm) }
    }
}

impl<T, S, D> Matrix<Owned<T>, S, D>
where
    T: Num,
    D: DeviceBase,
    S: DimTrait,
{
    pub fn to_ref_mut<'a>(&mut self) -> Matrix<Ref<&'a mut T>, S, D> {
        Matrix {
            ptr: self.ptr.to_ref_mut(),
            shape: self.shape,
            stride: self.stride,
        }
    }

    pub fn to<Dout: DeviceBase>(self) -> Matrix<Owned<T>, S, Dout> {
        let shape = self.shape();
        let stride = self.stride();
        let ptr = self.ptr.to::<Dout>();
        Matrix::new(ptr, shape, stride)
    }
}

impl<'a, T, S, D> Matrix<Ref<&'a mut T>, S, D>
where
    T: Num,
    D: DeviceBase,
    S: DimTrait,
{
    pub(crate) fn offset_ptr_mut(&self, offset: usize) -> Ptr<Ref<&'a mut T>, D> {
        self.ptr.clone().offset_ptr_mut(offset)
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        unsafe { self.ptr.ptr.add(self.offset()) }
    }

    #[expect(clippy::missing_panics_doc)]
    pub fn as_mut_slice(&self) -> &mut [T] {
        if self.shape().len() <= 1 {
            self.as_mut_slice_unchecked()
        } else {
            panic!("Invalid shape");
        }
    }

    #[expect(clippy::mut_from_ref)]
    pub fn as_mut_slice_unchecked(&self) -> &mut [T] {
        let num_elm = self.shape().num_elm();
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), num_elm) }
    }

    pub fn each_by<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T),
    {
        let num_elm = self.shape().num_elm();
        let mut ptr = self.as_mut_ptr();
        for _ in 0..num_elm {
            f(unsafe { &mut *ptr });
            ptr = unsafe { ptr.add(1) };
        }
    }

    #[must_use]
    pub fn slice_mut<I>(&self, index: I) -> Matrix<Ref<&'a mut T>, S, D>
    where
        I: SliceTrait<Dim = S>,
    {
        let shape = self.shape();
        let stride = self.stride();
        let new_shape_stride = index.sliced_shape_stride(shape, stride);
        let offset = index.sliced_offset(stride);
        Matrix {
            ptr: self.ptr.clone().offset_ptr_mut(offset),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
        }
    }

    pub fn slice_mut_dyn(&self, index: Slice) -> Matrix<Ref<&'a mut T>, DimDyn, D> {
        let shape_stride = self.shape_stride().into_dyn();
        let new_shape_stride =
            index.sliced_shape_stride(shape_stride.shape(), shape_stride.stride());
        let offset = index.sliced_offset(shape_stride.stride());
        Matrix {
            ptr: self.ptr.clone().offset_ptr_mut(offset),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
        }
    }

    #[must_use]
    pub fn index_axis_mut<I>(&self, index: I) -> Matrix<Ref<&'a mut T>, S, D>
    where
        I: IndexAxisTrait,
        S: LessDimTrait,
        S::LessDim: DimTrait,
    {
        let shape = self.shape();
        let stride = self.stride();
        let new_shape_stride = index.get_shape_stride(shape, stride);
        let offset = index.offset(stride);
        Matrix {
            ptr: self.ptr.clone().offset_ptr_mut(offset),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
        }
    }

    pub fn index_axis_mut_dyn<I>(&self, index: I) -> Matrix<Ref<&'a mut T>, DimDyn, D>
    where
        I: IndexAxisTrait,
    {
        let shape_stride = self.shape_stride().into_dyn();
        let new_shape_stride = index.get_shape_stride(shape_stride.shape(), shape_stride.stride());
        let offset = index.offset(shape_stride.stride());
        Matrix {
            ptr: self.ptr.clone().offset_ptr_mut(offset),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
        }
    }

    #[expect(clippy::missing_panics_doc)]
    pub fn index_item_assign<I: Into<S>>(&self, index: I, value: T) {
        let index = index.into();
        assert!(!self.shape().is_overflow(index), "Index out of bounds");
        let offset = cal_offset(index, self.stride());
        self.ptr.assign_item(offset, value);
    }
}

#[expect(clippy::float_cmp)]
#[cfg(test)]
mod matrix_test {

    use crate::{
        device::DeviceBase,
        dim::{Dim1, Dim2, DimDyn, DimTrait},
        index::Index0D,
        slice, slice_dynamic,
    };

    use super::{Matrix, Owned};

    fn index_item_1d<D: DeviceBase>() {
        let m: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1.0, 2.0, 3.0], [3]);
        assert_eq!(m.index_item([0]), 1.0);
        assert_eq!(m.index_item([1]), 2.0);
        assert_eq!(m.index_item([2]), 3.0);
    }
    #[test]
    fn index_item_1d_cpu() {
        index_item_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn index_item_1d_nvidia() {
        index_item_1d::<crate::device::nvidia::Nvidia>();
    }

    fn index_item_2d<D: DeviceBase>() {
        let m: Matrix<Owned<f32>, Dim2, D> = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        assert_eq!(m.index_item([0, 0]), 1.0);
        assert_eq!(m.index_item([0, 1]), 2.0);
        assert_eq!(m.index_item([1, 0]), 3.0);
        assert_eq!(m.index_item([1, 1]), 4.0);

        let m: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        assert_eq!(m.index_item([0, 0]), 1.0);
        assert_eq!(m.index_item([0, 1]), 2.0);
        assert_eq!(m.index_item([1, 0]), 3.0);
        assert_eq!(m.index_item([1, 1]), 4.0);
    }
    #[test]
    fn index_item_2d_cpu() {
        index_item_2d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn index_item_2d_nvidia() {
        index_item_2d::<crate::device::nvidia::Nvidia>();
    }

    #[expect(clippy::cast_precision_loss)]
    fn slice_1d<D: DeviceBase>() {
        let v = (1..10).map(|x| x as f32).collect::<Vec<f32>>();
        let m: Matrix<Owned<f32>, Dim1, D> = Matrix::from_vec(v.clone(), [9]);
        let s = m.slice(slice!(1..4));
        assert_eq!(s.shape().slice(), [3]);
        assert_eq!(s.stride().slice(), [1]);
        assert_eq!(s.index_item([0]), 2.0);
        assert_eq!(s.index_item([1]), 3.0);
        assert_eq!(s.index_item([2]), 4.0);
    }
    #[test]
    fn slice_1d_cpu() {
        slice_1d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn slice_1d_nvidia() {
        slice_1d::<crate::device::nvidia::Nvidia>();
    }

    #[expect(clippy::cast_precision_loss)]
    fn slice_2d<D: DeviceBase>() {
        let v = (1..13).map(|x| x as f32).collect::<Vec<f32>>();
        let m: Matrix<Owned<f32>, Dim2, D> = Matrix::from_vec(v.clone(), [3, 4]);
        let s = m.slice(slice!(1..3, 1..4));
        assert_eq!(s.shape().slice(), [2, 3]);
        assert_eq!(s.stride().slice(), [4, 1]);

        assert_eq!(s.index_item([0, 0]), 6.);
        assert_eq!(s.index_item([0, 1]), 7.);
        assert_eq!(s.index_item([0, 2]), 8.);
        assert_eq!(s.index_item([1, 0]), 10.);
        assert_eq!(s.index_item([1, 1]), 11.);
        assert_eq!(s.index_item([1, 2]), 12.);
    }
    #[test]
    fn slice_2d_cpu() {
        slice_2d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn slice_2d_nvidia() {
        slice_2d::<crate::device::nvidia::Nvidia>();
    }

    #[expect(clippy::cast_precision_loss)]
    fn slice_dyn_4d<D: DeviceBase>() {
        let v = (1..65).map(|x| x as f32).collect::<Vec<f32>>();
        let m: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(v.clone(), [2, 2, 4, 4]);
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
    #[test]
    fn slice_dyn_4d_cpu() {
        slice_dyn_4d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn slice_dyn_4d_nvidia() {
        slice_dyn_4d::<crate::device::nvidia::Nvidia>();
    }

    #[expect(clippy::cast_precision_loss)]
    fn index_axis_dyn_2d<D: DeviceBase>() {
        let v = (1..13).map(|x| x as f32).collect::<Vec<f32>>();
        let m: Matrix<Owned<f32>, DimDyn, D> = Matrix::from_vec(v.clone(), [3, 4]);
        let s = m.index_axis_dyn(Index0D::new(0));

        assert_eq!(s.index_item([0]), 1.);
        assert_eq!(s.index_item([1]), 2.);
        assert_eq!(s.index_item([2]), 3.);
    }
    #[test]
    fn index_axis_dyn_2d_cpu() {
        index_axis_dyn_2d::<crate::device::cpu::Cpu>();
    }
    #[cfg(feature = "nvidia")]
    #[test]
    fn index_axis_dyn_2d_nvidia() {
        index_axis_dyn_2d::<crate::device::nvidia::Nvidia>();
    }
}
