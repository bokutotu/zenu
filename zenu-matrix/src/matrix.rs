use std::marker::PhantomData;

use crate::{
    dim::{default_stride, DimTrait},
    index::SliceTrait,
    num::Num,
    shape_stride::ShapeStride,
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
    fn offset_ptr<T>(ptr: *const T, offset: isize) -> *const T;
    fn drop_ptr<T>(ptr: *mut T, len: usize);
    fn clone_ptr<T>(ptr: *const T, len: usize) -> *mut T;
    fn assign_item<T>(ptr: *mut T, offset: usize, value: T);
    fn get_item<T>(ptr: *const T, offset: usize) -> T;
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

impl<R, D> Drop for Ptr<R, D>
where
    R: Repr,
    D: Device,
{
    fn drop(&mut self) {
        R::drop_memory(self.ptr, self.len, D::default());
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

    pub fn stdide(&self) -> S {
        self.stride
    }

    pub fn is_default_stride(&self) -> bool {
        self.shape_stride().is_default_stride()
    }

    pub fn is_transpose_default_stride(&self) -> bool {
        self.shape_stride().is_transposed_default_stride()
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

pub trait ToOwned<R, S, D>
where
    R: Repr,
    S: DimTrait,
    D: Device,
{
    fn to_owned(&self) -> Matrix<Owned<R::Item>, S, D>;
}

impl<T, S, D> ToOwned<Ref<&T>, S, D> for Matrix<Ref<&T>, S, D>
where
    T: Num,
    S: DimTrait,
    D: Device,
{
    fn to_owned(&self) -> Matrix<Owned<T>, S, D> {
        let vec =
            unsafe { Vec::from_raw_parts(self.ptr.ptr as *mut T, self.ptr.len, self.ptr.len) };

        Matrix::from_vec(vec, self.shape)
    }
}

impl<T, S, D> ToOwned<Ref<&mut T>, S, D> for Matrix<Ref<&mut T>, S, D>
where
    T: Num,
    S: DimTrait,
    D: Device,
{
    fn to_owned(&self) -> Matrix<Owned<T>, S, D> {
        let vec =
            unsafe { Vec::from_raw_parts(self.ptr.ptr as *mut T, self.ptr.len, self.ptr.len) };

        Matrix::from_vec(vec, self.shape)
    }
}

pub trait AsMutPtr<T, D>
where
    D: Device,
{
    fn as_mut_ptr(self) -> *mut T;
}

impl<T, S, D> AsMutPtr<T, D> for &mut Matrix<Owned<T>, S, D>
where
    T: Num,
    S: DimTrait,
    D: Device,
{
    fn as_mut_ptr(self) -> *mut T {
        self.ptr.ptr
    }
}

impl<T, S, D> AsMutPtr<T, D> for Matrix<Ref<&mut T>, S, D>
where
    T: Num,
    S: DimTrait,
    D: Device,
{
    fn as_mut_ptr(self) -> *mut T {
        self.ptr.ptr
    }
}

pub trait AsPtr<T, D>
where
    T: Num,
    D: Device,
{
    fn as_ptr(self) -> *const T;
}

impl<T, S, D> AsPtr<T, D> for &Matrix<Owned<T>, S, D>
where
    T: Num,
    S: DimTrait,
    D: Device,
{
    fn as_ptr(self) -> *const T {
        self.ptr.ptr
    }
}

impl<T, S, D> AsPtr<T, D> for Matrix<Ref<&T>, S, D>
where
    T: Num,
    S: DimTrait,
    D: Device,
{
    fn as_ptr(self) -> *const T {
        self.ptr.ptr
    }
}

impl<T, S, D> AsPtr<T, D> for Matrix<Ref<&mut T>, S, D>
where
    T: Num,
    S: DimTrait,
    D: Device,
{
    fn as_ptr(self) -> *const T {
        self.ptr.ptr
    }
}

pub trait MatrixSlice<T, S, D, I>
where
    T: Num,
    S: DimTrait,
    D: Device,
    I: SliceTrait<Dim = S>,
{
    fn slice(&self, index: I) -> Matrix<Ref<&T>, S, D>;
}

pub trait MatrixSliceMut<T, S, D, I>
where
    T: Num,
    S: DimTrait,
    D: Device,
    I: SliceTrait<Dim = S>,
{
    fn slice_mut(&self, index: I) -> Matrix<Ref<&mut T>, S, D>;
}
