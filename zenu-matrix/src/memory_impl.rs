use crate::{
    cpu_blas::CpuBlas,
    cpu_element_wise::CpuElementWise,
    memory::{
        Memory, MemoryAccessor, Owned, ToOwnedMemory, ToViewMemory, ToViewMutMemory, View, ViewMut,
    },
};
use std::ptr::NonNull;

use crate::num::Num;

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuAccessor<T: Num> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Num> CpuAccessor<T> {
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Num> MemoryAccessor for CpuAccessor<T> {
    type Item = T;

    fn value(&self, ptr: NonNull<Self::Item>, offset: usize) -> Self::Item {
        unsafe { *ptr.as_ptr().add(offset) }
    }

    fn set_value(&mut self, ptr: NonNull<Self::Item>, offset: usize, value: Self::Item) {
        unsafe { *ptr.as_ptr().add(offset) = value };
    }

    fn clone_ptr(&self, ptr: NonNull<Self::Item>, len: usize) -> NonNull<Self::Item> {
        let vec = unsafe { Vec::from_raw_parts(ptr.as_ptr(), len, len) };
        let mut vec_c = vec.clone();
        std::mem::forget(vec);
        let ptr = NonNull::new(vec_c.as_mut_ptr()).unwrap();
        std::mem::forget(vec_c);
        ptr
    }

    fn drop(&self, ptr: *const Self::Item, len: usize) {
        let _ = unsafe { Vec::from_raw_parts(ptr as *mut T, len, len) };
    }

    fn offset_ptr(&self, ptr: NonNull<Self::Item>, offset: usize) -> NonNull<Self::Item> {
        NonNull::new(unsafe { ptr.as_ptr().add(offset) }).unwrap()
    }
}

#[derive(Debug)]
pub struct OwnedMem<T: Num> {
    ptr: NonNull<T>,
    offset: usize,
    length: usize,
    accessor: CpuAccessor<T>,
}

#[derive(Debug)]
pub struct ViewMem<'a, T: Num> {
    ptr: &'a OwnedMem<T>,
    offset: usize,
}

#[derive(Debug)]
pub struct ViewMutMem<'a, T: Num> {
    ptr: &'a mut OwnedMem<T>,
    offset: usize,
}

impl<T: Num> Memory for OwnedMem<T> {
    type Item = T;
    type Blas = CpuBlas<T>;
    type ElmentWise = CpuElementWise<T>;

    fn len(&self) -> usize {
        self.length
    }

    fn as_ptr(&self) -> *const Self::Item {
        self.ptr.as_ptr()
    }

    fn get_offset(&self) -> usize {
        self.offset
    }

    fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
    }

    fn as_ptr_offset(&self, offset: usize) -> *const Self::Item {
        self.accessor
            .offset_ptr(self.ptr, self.get_offset() + offset)
            .as_ptr()
    }

    fn value_offset(&self, offset: usize) -> Self::Item {
        self.accessor.value(self.ptr, self.get_offset() + offset)
    }
}

impl<T: Num> Clone for OwnedMem<T> {
    fn clone(&self) -> Self {
        let ptr = self.accessor.clone_ptr(self.ptr, self.len());
        Self {
            ptr,
            offset: self.offset,
            length: self.length,
            accessor: self.accessor,
        }
    }
}

impl<T> ToViewMemory for OwnedMem<T>
where
    T: Num,
{
    type View<'a> = ViewMem<'a, T>
    where
        Self: 'a;

    fn to_view(&self, offset: usize) -> Self::View<'_> {
        ViewMem { ptr: self, offset }
    }
}

impl<T> ToViewMutMemory for OwnedMem<T>
where
    T: Num,
{
    type ViewMut<'a> = ViewMutMem<'a, T>
    where
        Self: 'a;

    fn to_view_mut(&mut self, offset: usize) -> Self::ViewMut<'_> {
        ViewMutMem { ptr: self, offset }
    }
}

impl<'a, T> ToViewMutMemory for ViewMutMem<'a, T>
where
    T: Num,
{
    type ViewMut<'b> = ViewMutMem<'b, T>
    where
        Self: 'b;

    fn to_view_mut(&mut self, offset: usize) -> Self::ViewMut<'_> {
        let offset = self.get_offset() + offset;
        ViewMutMem {
            ptr: self.ptr,
            offset,
        }
    }
}

impl<T: Num> ToOwnedMemory for OwnedMem<T> {
    type Owned = OwnedMem<T>;

    fn to_owned_memory(&self) -> Self::Owned {
        self.clone()
    }
}

impl<T: Num> Owned for OwnedMem<T> {
    fn from_vec(vec: Vec<Self::Item>) -> Self {
        let ptr = unsafe { NonNull::new_unchecked(vec.as_ptr() as *mut T) };
        let length = vec.len();
        std::mem::forget(vec);
        Self {
            ptr,
            offset: 0,
            length,
            accessor: CpuAccessor::new(),
        }
    }
}

impl<T: Num> Drop for OwnedMem<T> {
    fn drop(&mut self) {
        self.accessor.drop(self.ptr.as_ptr(), self.len());
    }
}

impl<T: Num> Clone for ViewMem<'_, T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            offset: self.offset,
        }
    }
}

macro_rules! impl_cpu_memory_to_view {
    ($impl_ty: ty) => {
        impl<'a, T: Num> Memory for $impl_ty {
            type Item = T;
            type Blas = CpuBlas<T>;
            type ElmentWise = CpuElementWise<T>;

            fn len(&self) -> usize {
                self.ptr.len()
            }

            fn as_ptr(&self) -> *const Self::Item {
                self.ptr.as_ptr()
            }

            fn get_offset(&self) -> usize {
                self.offset
            }

            fn set_offset(&mut self, offset: usize) {
                self.offset = offset - self.ptr.get_offset();
            }

            fn as_ptr_offset(&self, offset: usize) -> *const Self::Item {
                self.ptr.as_ptr_offset(self.get_offset() + offset)
            }

            fn value_offset(&self, offset: usize) -> Self::Item {
                self.ptr.value_offset(self.get_offset() + offset)
            }
        }

        impl<'a, T> ToViewMemory for $impl_ty
        where
            T: Num,
        {
            type View<'b> = ViewMem<'b, T> where Self: 'b;

            fn to_view(&self, offset: usize) -> Self::View<'_> {
                ViewMem {
                    ptr: self.ptr,
                    offset: self.get_offset() + offset,
                }
            }
        }

        impl<'a, T> ToOwnedMemory for $impl_ty
        where
            T: Num,
        {
            type Owned = OwnedMem<T>;

            fn to_owned_memory(&self) -> Self::Owned {
                let mut memory = self.ptr.clone();
                memory.set_offset(self.offset + self.ptr.get_offset());
                memory
            }
        }
    };
}
impl_cpu_memory_to_view!(ViewMem<'a, T>);
impl_cpu_memory_to_view!(ViewMutMem<'a, T>);

impl<'a, T: Num> View for ViewMem<'a, T> {}
impl<'a, T: Num> ViewMut for ViewMutMem<'a, T> {
    fn as_mut_ptr(&self) -> *mut Self::Item {
        self.ptr.as_ptr() as *mut Self::Item
    }
}

#[cfg(test)]
mod memory {
    use super::*;

    use crate::memory::Memory;

    #[test]
    fn owned_memory_offset() {
        let mut memory = OwnedMem::from_vec(vec![1., 2., 3., 4., 5.]);
        assert_eq!(memory.get_offset(), 0);
        assert_eq!(memory.value_offset(0), 1.);

        memory.set_offset(2);
        assert_eq!(memory.get_offset(), 2);
        assert_eq!(memory.value_offset(0), 3.);
    }

    #[test]
    fn view_memory_offset_without_owned_memory_offset() {
        let memory = OwnedMem::from_vec(vec![1., 2., 3., 4., 5.]);
        let view = memory.to_view(3);
        assert_eq!(view.get_offset(), 3);
        assert_eq!(view.value_offset(0), 4.);
    }

    #[test]
    fn view_memory_offset_with_owned_memory_offset() {
        let mut memory = OwnedMem::from_vec(vec![1., 2., 3., 4., 5.]);
        memory.set_offset(1);
        let view = memory.to_view(2);
        assert_eq!(view.get_offset(), 2);
        assert_eq!(view.value_offset(0), 4.);
    }

    #[test]
    fn owned_memory_view_memory_owned_memory() {
        let mut memory = OwnedMem::from_vec(vec![1., 2., 3., 4., 5.]);
        memory.set_offset(1);
        let view = memory.to_view(3);
        let owned_memory = view.to_owned_memory();
        assert_eq!(owned_memory.get_offset(), 4);
        assert_eq!(owned_memory.value_offset(0), 5.);
    }
}