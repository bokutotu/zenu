use std::marker::PhantomData;

use crate::{
    device::DeviceBase,
    dim::{cal_offset, default_stride, DimDyn, DimTrait, LessDimTrait},
    index::{IndexAxisTrait, SliceTrait},
    num::Num,
    shape_stride::ShapeStride,
    slice::Slice,
};

pub trait Repr: Default + Clone {
    type Item: Num;

    fn drop_memory<D: DeviceBase>(ptr: *mut Self::Item, len: usize, _: D);
    fn clone_memory<D: DeviceBase>(ptr: *mut Self::Item, len: usize, _: D) -> *mut Self::Item;
    fn to_ref(&self) -> Ref<&Self::Item>;
}

pub trait OwnedRepr: Repr {}
pub trait ToRefMut: Repr {
    fn to_ref_mut(&mut self) -> Ref<&mut Self::Item>;
}

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
    fn to_ref(&self) -> Ref<&'a Self::Item> {
        Ref {
            _maker: PhantomData,
        }
    }
}

impl<'a, T: Num> Repr for Ref<&'a mut T> {
    type Item = T;

    fn drop_memory<D: DeviceBase>(_ptr: *mut Self::Item, _len: usize, _: D) {}
    fn clone_memory<D: DeviceBase>(ptr: *mut Self::Item, _len: usize, _: D) -> *mut Self::Item {
        ptr
    }
    fn to_ref(&self) -> Ref<&'a Self::Item> {
        Ref {
            _maker: PhantomData,
        }
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
    fn to_ref(&self) -> Ref<&Self::Item> {
        Ref {
            _maker: PhantomData,
        }
    }
}

impl<T: Num> Clone for Owned<T> {
    fn clone(&self) -> Self {
        Owned {
            _maker: PhantomData,
        }
    }
}

impl<'a, T: Num> Clone for Ref<&'a T> {
    fn clone(&self) -> Self {
        Ref {
            _maker: PhantomData,
        }
    }
}

impl<'a, T: Num> Clone for Ref<&'a mut T> {
    fn clone(&self) -> Self {
        Ref {
            _maker: PhantomData,
        }
    }
}

impl<T: Num> OwnedRepr for Owned<T> {}

impl<T: Num> ToRefMut for Owned<T> {
    fn to_ref_mut(&mut self) -> Ref<&mut T> {
        Ref {
            _maker: PhantomData,
        }
    }
}

pub struct Matrix<R, S, D>
where
    R: Repr,
    S: DimTrait,
    D: DeviceBase,
{
    ptr: *mut R::Item,
    len: usize,
    offset: usize,
    repr: R,
    shape: S,
    stride: S,
    device: D,
}

impl<T, S, D> Clone for Matrix<Ref<&T>, S, D>
where
    T: Num,
    S: DimTrait,
    D: DeviceBase,
{
    fn clone(&self) -> Self {
        Matrix {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset,
            repr: self.repr.clone(),
            shape: self.shape,
            stride: self.stride,
            device: D::default(),
        }
    }
}

impl<T, S, D> Clone for Matrix<Ref<&mut T>, S, D>
where
    T: Num,
    S: DimTrait,
    D: DeviceBase,
{
    fn clone(&self) -> Self {
        Matrix {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset,
            repr: self.repr.clone(),
            shape: self.shape,
            stride: self.stride,
            device: D::default(),
        }
    }
}

impl<T, S, D> Clone for Matrix<Owned<T>, S, D>
where
    T: Num,
    S: DimTrait,
    D: DeviceBase,
{
    fn clone(&self) -> Self {
        let ptr = Owned::<T>::clone_memory(self.ptr, self.len, self.device);
        Matrix {
            ptr,
            len: self.len,
            offset: self.offset,
            repr: self.repr.clone(),
            shape: self.shape,
            stride: self.stride,
            device: self.device,
        }
    }
}

impl<R, S, D> Drop for Matrix<R, S, D>
where
    R: Repr,
    S: DimTrait,
    D: DeviceBase,
{
    fn drop(&mut self) {
        R::drop_memory(self.ptr, self.len, self.device);
    }
}

impl<R, S, D> Matrix<R, S, D>
where
    R: Repr,
    S: DimTrait,
    D: DeviceBase,
{
    fn from_raw(ptr: *mut R::Item, len: usize, offset: usize, shape: S, stride: S) -> Self {
        Matrix {
            ptr,
            len,
            repr: R::default(),
            offset,
            device: D::default(),
            shape,
            stride,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
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

    pub fn update_offset(&mut self, offset: usize) {
        self.offset += offset;
    }

    pub fn is_default_stride(&self) -> bool {
        self.shape_stride().is_default_stride()
    }

    pub fn is_transpose_default_stride(&self) -> bool {
        self.shape_stride().is_transposed_default_stride()
    }

    pub fn as_ptr(&self) -> *const R::Item {
        unsafe { self.ptr.add(self.offset()) }
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
            len: self.len,
            offset: self.offset,
            repr: self.repr.clone(),
            shape,
            stride,
            device: self.device,
        }
    }

    pub fn update_shape_stride(&mut self, shape: S, stride: S) {
        self.shape = shape;
        self.stride = stride;
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
            len: self.len,
            offset: self.offset,
            repr: self.repr.clone(),
            device: self.device,
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
            ptr: self.ptr,
            len: self.len,
            offset: self.offset + offset,
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
            repr: self.repr.to_ref(),
            device: self.device,
        }
    }

    pub fn slice_dyn(&self, index: Slice) -> Matrix<Ref<&R::Item>, DimDyn, D> {
        let shape_stride = self.shape_stride().into_dyn();
        let new_shape_stride =
            index.sliced_shape_stride(shape_stride.shape(), shape_stride.stride());
        let offset = index.sliced_offset(shape_stride.stride());
        Matrix {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset + offset,
            repr: self.repr.to_ref(),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
            device: self.device,
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
            ptr: self.ptr,
            len: self.len,
            offset: self.offset + offset,
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
            repr: self.repr.to_ref(),
            device: self.device,
        }
    }

    pub fn index_axis_dyn<I>(&self, index: I) -> Matrix<Ref<&R::Item>, DimDyn, D>
    where
        I: IndexAxisTrait,
    {
        let mut rf = self.to_ref().into_dyn_dim();
        let shape_stride = rf.shape_stride().into_dyn();
        let new_shape_stride = index.get_shape_stride(shape_stride.shape(), shape_stride.stride());
        let offset = index.offset(shape_stride.stride());
        rf.shape = new_shape_stride.shape();
        rf.stride = new_shape_stride.stride();
        rf.update_offset(offset);
        rf
    }

    pub fn index_item<I: Into<S>>(&self, index: I) -> R::Item {
        let index = index.into();
        if self.shape().is_overflow(index) {
            panic!("Index out of bounds");
        }
        let offset = cal_offset(index, self.stride());
        D::get_item(self.ptr as *const R::Item, offset)
    }

    pub fn to_ref(&self) -> Matrix<Ref<&R::Item>, S, D> {
        Matrix {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset,
            repr: self.repr.to_ref(),
            shape: self.shape,
            stride: self.stride,
            device: self.device,
        }
    }

    pub fn convert_dim_type<Dout: DimTrait>(self) -> Matrix<R, Dout, D> {
        Matrix {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset,
            repr: self.repr.clone(),
            device: self.device,
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

        let ptr = D::from_vec(vec);

        let stride = default_stride(shape);
        Matrix {
            ptr,
            len,
            offset: 0,
            repr: Owned::<T>::default(),
            shape,
            stride,
            device: D::default(),
        }
    }

    pub fn to_ref_mut(&mut self) -> Matrix<Ref<&mut T>, S, D> {
        Matrix {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset,
            repr: self.repr.to_ref_mut(),
            shape: self.shape,
            stride: self.stride,
            device: self.device,
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
        unsafe { self.ptr.add(self.offset()) }
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
            ptr: self.ptr,
            len: self.len,
            offset: self.offset + offset,
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
            repr: self.repr.clone(),
            device: self.device,
        }
    }

    pub fn slice_mut_dyn(&self, index: Slice) -> Matrix<Ref<&'a mut T>, DimDyn, D> {
        let shape_stride = self.shape_stride().into_dyn();
        let new_shape_stride =
            index.sliced_shape_stride(shape_stride.shape(), shape_stride.stride());
        let offset = index.sliced_offset(shape_stride.stride());
        Matrix {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset + offset,
            repr: self.repr.clone(),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
            device: self.device,
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
            ptr: self.ptr,
            len: self.len,
            offset: self.offset + offset,
            repr: self.repr.clone(),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
            device: self.device,
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
            ptr: self.ptr,
            len: self.len,
            offset: self.offset + offset,
            repr: self.repr.clone(),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
            device: self.device,
        }
    }

    pub fn index_item_assign<I: Into<S>>(&mut self, index: I, value: T) {
        let index = index.into();
        if self.shape().is_overflow(index) {
            panic!("Index out of bounds");
        }
        let offset = cal_offset(index, self.stride());
        D::assign_item(self.as_mut_ptr(), offset, value);
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
