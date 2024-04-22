use std::marker::PhantomData;

use crate::{dim::DimTrait, num::Num};

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
    stdide: S,
}

impl<R, S, D> Matrix<R, S, D>
where
    R: Repr,
    S: DimTrait,
    D: Device,
{
    pub fn to_ref(&self) -> Matrix<Ref<&R::Item>, S, D>
    where
        R: OwnedRepr,
    {
        Matrix {
            ptr: self.ptr.to_ref(),
            shape: self.shape,
            stdide: self.stdide,
        }
    }

    pub fn to_ref_mut(&mut self) -> Matrix<Ref<&mut R::Item>, S, D>
    where
        R: OwnedRepr,
    {
        Matrix {
            ptr: self.ptr.to_ref_mut(),
            shape: self.shape,
            stdide: self.stdide,
        }
    }
}
