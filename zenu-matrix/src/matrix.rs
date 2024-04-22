use std::marker::PhantomData;

use crate::{
    dim::{cal_offset, default_stride, DimDyn, DimTrait, LessDimTrait},
    index::{IndexAxisTrait, SliceTrait},
    num::Num,
    shape_stride::ShapeStride,
    slice::Slice,
};

pub trait Repr: Default {
    type Item: Num;

    fn drop_memory<D: Device>(ptr: *mut Self::Item, len: usize, _: D);
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

    fn drop_memory<D: Device>(ptr: *mut Self::Item, len: usize, _: D) {
        D::drop_ptr(ptr, len);
    }
}

impl<'a, T: Num> Repr for Ref<&'a mut T> {
    type Item = T;

    fn drop_memory<D: Device>(_ptr: *mut Self::Item, _len: usize, _: D) {}
}

impl<T: Num> Repr for Owned<T> {
    type Item = T;

    fn drop_memory<D: Device>(_ptr: *mut Self::Item, _len: usize, _: D) {}
}

impl<T: Num> OwnedRepr for Owned<T> {}

pub trait Device: Copy + Default {
    fn drop_ptr<T>(ptr: *mut T, len: usize);
    fn clone_ptr<T>(ptr: *const T, len: usize) -> *mut T;
    fn assign_item<T: Num>(ptr: *mut T, offset: usize, value: T);
    fn get_item<T: Num>(ptr: *const T, offset: usize) -> T;
}

#[derive(Copy, Clone, Default)]
pub struct Cpu;

impl Device for Cpu {
    fn drop_ptr<T>(ptr: *mut T, len: usize) {
        unsafe {
            std::vec::Vec::from_raw_parts(ptr, 0, len);
        }
    }

    fn clone_ptr<T>(ptr: *const T, len: usize) -> *mut T {
        let mut vec = Vec::with_capacity(len);
        unsafe {
            vec.set_len(len);
            std::ptr::copy_nonoverlapping(ptr, vec.as_mut_ptr(), len);
        }
        vec.as_mut_ptr()
    }

    fn assign_item<T>(ptr: *mut T, offset: usize, value: T) {
        unsafe {
            ptr.add(offset).write(value);
        }
    }

    fn get_item<T>(ptr: *const T, offset: usize) -> T {
        unsafe { ptr.add(offset).read() }
    }
}

#[cfg(feature = "nvidia")]
#[derive(Copy, Clone, Default)]
pub struct Nvidia;

#[cfg(feature = "nvidia")]
impl Device for Nvidia {
    fn drop_ptr<T>(ptr: *mut T, _: usize) {
        zenu_cuda::runtime::cuda_free(ptr as *mut std::ffi::c_void).unwrap();
    }

    fn clone_ptr<T>(src: *const T, len: usize) -> *mut T {
        let dst = zenu_cuda::runtime::cuda_malloc(len).unwrap() as *mut T;
        zenu_cuda::runtime::cuda_copy(
            dst,
            src,
            len,
            zenu_cuda::runtime::ZenuCudaMemCopyKind::HostToHost,
        )
        .unwrap();
        dst
    }

    fn assign_item<T: Num>(ptr: *mut T, offset: usize, value: T) {
        zenu_cuda::kernel::set_memory(ptr, offset, value);
    }

    fn get_item<T: Num>(ptr: *const T, offset: usize) -> T {
        zenu_cuda::kernel::get_memory(ptr, offset)
    }
}

pub struct Ptr<R, D>
where
    R: Repr,
    D: Device,
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
    D: Device,
{
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
}

impl<R, D> Drop for Ptr<R, D>
where
    R: Repr,
    D: Device,
{
    fn drop(&mut self) {
        R::drop_memory(self.ptr, self.len, D::default());
    }
}

impl<T: Num, D: Device> Ptr<Ref<&mut T>, D> {
    pub fn offset_ptr_mut(&self, offset: usize) -> Ptr<Ref<&mut T>, D> {
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

impl<R: OwnedRepr, D: Device> Clone for Ptr<R, D> {
    fn clone(&self) -> Self {
        let ptr = D::clone_ptr(self.ptr, self.len);
        Ptr {
            ptr,
            len: self.len,
            offset: self.offset,
            repr: PhantomData,
            device: PhantomData,
        }
    }
}

impl<T: Num, D: Device> Clone for Ptr<Ref<&T>, D> {
    fn clone(&self) -> Self {
        Ptr {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset,
            repr: PhantomData,
            device: PhantomData,
        }
    }
}

impl<T: Num, D: Device> Clone for Ptr<Ref<&mut T>, D> {
    fn clone(&self) -> Self {
        Ptr {
            ptr: self.ptr,
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
    D: Device,
{
    fn to_ref(&self) -> Ptr<Ref<&R::Item>, D> {
        Ptr {
            ptr: self.ptr,
            len: self.len,
            offset: self.offset,
            repr: PhantomData,
            device: PhantomData,
        }
    }

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
    D: Device,
{
    ptr: Ptr<R, D>,
    shape: S,
    stride: S,
}

impl<R, S, D> Matrix<R, S, D>
where
    R: Repr,
    S: DimTrait,
    D: Device,
{
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
        self.ptr.ptr
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
}

impl<T, S, D> Matrix<Owned<T>, S, D>
where
    T: Num,
    D: Device,
    S: DimTrait,
{
    pub fn from_vec(mut vec: Vec<T>, shape: S) -> Self {
        if vec.len() != shape.num_elm() {
            panic!("Invalid size");
        }

        let len = vec.len();

        let ptr = Ptr {
            ptr: vec.as_mut_ptr(),
            len,
            offset: 0,
            repr: PhantomData,
            device: PhantomData,
        };

        std::mem::forget(vec);

        let stride = default_stride(shape);
        Matrix { ptr, shape, stride }
    }

    pub fn to_ref(&self) -> Matrix<Ref<&T>, S, D> {
        Matrix {
            ptr: self.ptr.to_ref(),
            shape: self.shape,
            stride: self.stride,
        }
    }

    pub fn to_ref_mut(&mut self) -> Matrix<Ref<&mut T>, S, D> {
        Matrix {
            ptr: self.ptr.to_ref_mut(),
            shape: self.shape,
            stride: self.stride,
        }
    }
}

impl<T, S, D> Matrix<Ref<&mut T>, S, D>
where
    T: Num,
    D: Device,
    S: DimTrait,
{
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr.ptr
    }

    pub fn as_mut_slice(&self) -> &mut [T] {
        if self.shape().len() <= 1 {
            let num_elm = std::cmp::max(self.shape().num_elm(), 1);
            unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), num_elm) }
        } else {
            panic!("Invalid shape");
        }
    }

    pub fn slice_mut<I>(&self, index: I) -> Matrix<Ref<&mut T>, S, D>
    where
        I: SliceTrait<Dim = S>,
    {
        let shape = self.shape();
        let stride = self.stride();
        let new_shape_stride = index.sliced_shape_stride(shape, stride);
        let offset = index.sliced_offset(stride);
        Matrix {
            ptr: self.ptr.offset_ptr_mut(offset),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
        }
    }

    pub fn slice_mut_dyn(&self, index: Slice) -> Matrix<Ref<&mut T>, DimDyn, D> {
        let shape_stride = self.shape_stride().into_dyn();
        let new_shape_stride =
            index.sliced_shape_stride(shape_stride.shape(), shape_stride.stride());
        let offset = index.sliced_offset(shape_stride.stride());
        Matrix {
            ptr: self.ptr.offset_ptr_mut(offset),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
        }
    }

    pub fn index_axis_mut<I>(&self, index: I) -> Matrix<Ref<&mut T>, S, D>
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
            ptr: self.ptr.offset_ptr_mut(offset),
            shape: new_shape_stride.shape(),
            stride: new_shape_stride.stride(),
        }
    }

    pub fn index_axis_mut_dyn<I>(&self, index: I) -> Matrix<Ref<&mut T>, DimDyn, D>
    where
        I: IndexAxisTrait,
    {
        let shape_stride = self.shape_stride().into_dyn();
        let new_shape_stride = index.get_shape_stride(shape_stride.shape(), shape_stride.stride());
        let offset = index.offset(shape_stride.stride());
        Matrix {
            ptr: self.ptr.offset_ptr_mut(offset),
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
