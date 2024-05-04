use std::marker::PhantomData;

use crate::{
    device::DeviceBase,
    dim::{cal_offset, default_stride, DimDyn, DimTrait, LessDimTrait},
    index::{IndexAxisTrait, SliceTrait},
    num::Num,
    shape_stride::ShapeStride,
    slice::Slice,
};

pub trait Repr: Default {
    type Item: Num;

    fn drop_memory<D: DeviceBase>(ptr: *mut Self::Item, len: usize, _: D);
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

    fn drop_memory<D: DeviceBase>(_ptr: *mut Self::Item, _len: usize, _: D) {}
    fn clone_memory<D: DeviceBase>(ptr: *mut Self::Item, _len: usize, _: D) -> *mut Self::Item {
        ptr
    }
}

impl<'a, T: Num> Repr for Ref<&'a mut T> {
    type Item = T;

    fn drop_memory<D: DeviceBase>(_ptr: *mut Self::Item, _len: usize, _: D) {}
    fn clone_memory<D: DeviceBase>(ptr: *mut Self::Item, _len: usize, _: D) -> *mut Self::Item {
        ptr
    }
}

impl<T: Num> Repr for Owned<T> {
    type Item = T;

    fn drop_memory<D: DeviceBase>(ptr: *mut Self::Item, len: usize, _: D) {
        D::drop_ptr(ptr, len);
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
    pub fn new(ptr: *mut R::Item, len: usize, offset: usize) -> Self {
        Ptr {
            ptr,
            len,
            offset,
            repr: PhantomData,
            device: PhantomData,
        }
    }

    pub fn offset_ptr(&self, offset: usize) -> Ptr<Ref<&R::Item>, D> {
        Ptr {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset + offset,
            repr: PhantomData,
            device: PhantomData,
        }
    }

    pub fn get_item(&self, offset: usize) -> R::Item {
        if offset >= self.len {
            panic!("Index out of bounds");
        }
        D::get_item(self.ptr, offset + self.offset)
    }

    fn to_ref(&self) -> Ptr<Ref<&R::Item>, D> {
        Ptr {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset,
            repr: PhantomData,
            device: PhantomData,
        }
    }
}

impl<R, D> Drop for Ptr<R, D>
where
    R: Repr,
    D: DeviceBase,
{
    fn drop(&mut self) {
        R::drop_memory(self.ptr, self.len, D::default());
    }
}

impl<'a, T: Num, D: DeviceBase> Ptr<Ref<&'a mut T>, D> {
    pub fn offset_ptr_mut(self, offset: usize) -> Ptr<Ref<&'a mut T>, D> {
        Ptr {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset + offset,
            repr: PhantomData,
            device: PhantomData,
        }
    }

    pub fn assign_item(&self, offset: usize, value: T) {
        if offset >= self.len {
            panic!("Index out of bounds");
        }
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
    fn to_ref_mut(&mut self) -> Ptr<Ref<&mut R::Item>, D> {
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
    pub fn new(ptr: Ptr<R, D>, shape: S, stride: S) -> Self {
        Matrix { ptr, shape, stride }
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

    pub fn as_slice(&self) -> &[R::Item] {
        if self.shape().len() <= 1 {
            let num_elm = std::cmp::max(self.shape().num_elm(), 1);
            unsafe { std::slice::from_raw_parts(self.as_ptr(), num_elm) }
        } else {
            panic!("Invalid shape");
        }
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

    pub fn index_item<I: Into<S>>(&self, index: I) -> R::Item {
        let index = index.into();
        if self.shape().is_overflow(index) {
            panic!("Index out of bounds");
        }
        let offset = cal_offset(index, self.stride());
        self.ptr.get_item(offset)
    }

    pub fn to_ref(&self) -> Matrix<Ref<&R::Item>, S, D> {
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
}

impl<T, S, D> Matrix<Owned<T>, S, D>
where
    T: Num,
    D: DeviceBase,
    S: DimTrait,
{
    pub fn from_vec<I: Into<S>>(vec: Vec<T>, shape: I) -> Self {
        let shape = shape.into();
        if vec.len() != shape.num_elm() {
            panic!("Invalid size");
        }

        let len = vec.len();

        let ptr = Ptr {
            ptr: D::from_vec(vec.clone()),
            len,
            offset: 0,
            repr: PhantomData,
            device: PhantomData,
        };

        std::mem::forget(vec);

        let stride = default_stride(shape);
        Matrix { ptr, shape, stride }
    }

    pub fn to_ref_mut(&mut self) -> Matrix<Ref<&mut T>, S, D> {
        Matrix {
            ptr: self.ptr.to_ref_mut(),
            shape: self.shape,
            stride: self.stride,
        }
    }
}

impl<'a, T, S, D> Matrix<Ref<&'a mut T>, S, D>
where
    T: Num,
    D: DeviceBase,
    S: DimTrait,
{
    pub fn as_mut_ptr(&self) -> *mut T {
        unsafe { self.ptr.ptr.add(self.offset()) }
    }

    pub fn as_mut_slice(&self) -> &mut [T] {
        if self.shape().len() <= 1 {
            let num_elm = std::cmp::max(self.shape().num_elm(), 1);
            unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), num_elm) }
        } else {
            panic!("Invalid shape");
        }
    }

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

    pub fn index_item_assign<I: Into<S>>(&mut self, index: I, value: T) {
        let index = index.into();
        if self.shape().is_overflow(index) {
            panic!("Index out of bounds");
        }
        let offset = cal_offset(index, self.stride());
        self.ptr.assign_item(offset, value);
    }
}

#[cfg(test)]
mod matrix {
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
