use crate::{
    cpu_blas::CpuBlas,
    cpu_element_wise::CpuElementWise,
    memory::{
        Memory, OwnedMemory, ToOwnedMemory, ToViewMemory, ToViewMutMemory, ViewMemory,
        ViewMutMemory,
    },
};
use std::ptr::NonNull;

use crate::num::Num;

#[derive(Debug)]
pub struct CpuOwnedMemory<T: Num> {
    ptr: NonNull<T>,
    offset: usize,
    length: usize,
}

#[derive(Debug)]
pub struct CpuViewMemory<'a, T: Num> {
    ptr: &'a CpuOwnedMemory<T>,
    offset: usize,
}

#[derive(Debug)]
pub struct CpuViewMutMemory<'a, T: Num> {
    ptr: &'a mut CpuOwnedMemory<T>,
    offset: usize,
}

impl<T: Num> Memory for CpuOwnedMemory<T> {
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
}

impl<T: Num> Clone for CpuOwnedMemory<T> {
    fn clone(&self) -> Self {
        let vec = unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len(), self.len()) };
        let mut vec_c = vec.clone();
        std::mem::forget(vec);
        let ptr = NonNull::new(vec_c.as_mut_ptr()).unwrap();
        std::mem::forget(vec_c);
        Self {
            ptr,
            offset: self.offset,
            length: self.length,
        }
    }
}

impl<T> ToViewMemory for CpuOwnedMemory<T>
where
    T: Num,
{
    type View<'a> = CpuViewMemory<'a, T>
    where
        Self: 'a;

    fn to_view(&self, offset: usize) -> Self::View<'_> {
        CpuViewMemory { ptr: self, offset }
    }
}

impl<T> ToViewMutMemory for CpuOwnedMemory<T>
where
    T: Num,
{
    type ViewMut<'a> = CpuViewMutMemory<'a, T>
    where
        Self: 'a;

    fn to_view_mut(&mut self, offset: usize) -> Self::ViewMut<'_> {
        CpuViewMutMemory { ptr: self, offset }
    }
}

impl<'a, T> ToViewMutMemory for CpuViewMutMemory<'a, T>
where
    T: Num,
{
    type ViewMut<'b> = CpuViewMutMemory<'b, T>
    where
        Self: 'b;

    fn to_view_mut(&mut self, offset: usize) -> Self::ViewMut<'_> {
        let offset = self.get_offset() + offset;
        CpuViewMutMemory {
            ptr: self.ptr,
            offset,
        }
    }
}

impl<T: Num> OwnedMemory for CpuOwnedMemory<T> {
    fn from_vec(vec: Vec<Self::Item>) -> Self {
        let ptr = unsafe { NonNull::new_unchecked(vec.as_ptr() as *mut T) };
        let length = vec.len();
        std::mem::forget(vec);
        Self {
            ptr,
            offset: 0,
            length,
        }
    }
}

impl<T: Num> Drop for CpuOwnedMemory<T> {
    fn drop(&mut self) {
        let _ = unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len(), self.len()) };
    }
}

impl<T: Num> Clone for CpuViewMemory<'_, T> {
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
                self.offset + self.ptr.get_offset()
            }

            fn set_offset(&mut self, offset: usize) {
                self.offset = offset - self.ptr.get_offset();
            }
        }

        impl<'a, T> ToViewMemory for $impl_ty
        where
            T: Num,
        {
            type View<'b> = CpuViewMemory<'b, T> where Self: 'b;

            fn to_view(&self, offset: usize) -> Self::View<'_> {
                CpuViewMemory {
                    ptr: self.ptr,
                    offset: self.get_offset() + offset,
                }
            }
        }

        impl<'a, T> ToOwnedMemory for $impl_ty
        where
            T: Num,
        {
            type Owned = CpuOwnedMemory<T>;

            fn to_owned_memory(&self) -> Self::Owned {
                let mut memory = self.ptr.clone();
                memory.set_offset(self.offset + self.ptr.get_offset());
                memory
            }
        }
    };
}
impl_cpu_memory_to_view!(CpuViewMemory<'a, T>);
impl_cpu_memory_to_view!(CpuViewMutMemory<'a, T>);

impl<'a, T: Num> ViewMemory for CpuViewMemory<'a, T> {}
impl<'a, T: Num> ViewMutMemory for CpuViewMutMemory<'a, T> {
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
        let mut memory = CpuOwnedMemory::from_vec(vec![1., 2., 3., 4., 5.]);
        assert_eq!(memory.get_offset(), 0);
        assert_eq!(memory.value_offset(0), 1.);

        memory.set_offset(2);
        assert_eq!(memory.get_offset(), 2);
        assert_eq!(memory.value_offset(0), 3.);
    }

    #[test]
    fn view_memory_offset_without_owned_memory_offset() {
        let memory = CpuOwnedMemory::from_vec(vec![1., 2., 3., 4., 5.]);
        let view = memory.to_view(3);
        assert_eq!(view.get_offset(), 3);
        assert_eq!(view.value_offset(0), 4.);
    }

    #[test]
    fn view_memory_offset_with_owned_memory_offset() {
        let mut memory = CpuOwnedMemory::from_vec(vec![1., 2., 3., 4., 5.]);
        memory.set_offset(1);
        let view = dbg!(memory.to_view(2));
        assert_eq!(view.get_offset(), 3);
        assert_eq!(view.value_offset(0), 4.);
    }

    #[test]
    fn owned_memory_view_memory_owned_memory() {
        let mut memory = CpuOwnedMemory::from_vec(vec![1., 2., 3., 4., 5.]);
        memory.set_offset(1);
        let view = memory.to_view(3);
        let owned_memory = view.to_owned_memory();
        assert_eq!(owned_memory.get_offset(), 4);
        assert_eq!(owned_memory.value_offset(0), 5.);
    }
}
