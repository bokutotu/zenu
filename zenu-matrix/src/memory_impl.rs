use serde::{Deserialize, Serialize};

use crate::{
    cpu_blas::CpuBlas,
    cpu_element_wise::CpuElementWise,
    memory::{
        Memory, MemoryAccessor, Owned, ToOwnedMemory, ToViewMemory, ToViewMutMemory, View, ViewMut,
    },
};
use std::ptr::NonNull;

use crate::num::Num;

#[cfg(feature = "nvidia")]
use zenu_cuda::{kernel::*, runtime::*};

#[derive(Clone, Copy, Debug, Default)]
pub struct Cpu<T: Num> {
    phantom: std::marker::PhantomData<T>,
}

impl<T: Num> Cpu<T> {
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "nvidia")]
#[derive(Clone, Copy, Debug, Default)]
pub struct Nvidia<T: Num> {
    phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "nvidia")]
impl<T: Num> Nvidia<T> {
    pub fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Num> MemoryAccessor for Cpu<T> {
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

#[cfg(feature = "nvidia")]
impl<T: Num> MemoryAccessor for Nvidia<T> {
    type Item = T;

    fn value(&self, ptr: NonNull<Self::Item>, offset: usize) -> Self::Item {
        get_memory(ptr.as_ptr(), offset)
    }

    fn set_value(&mut self, ptr: NonNull<Self::Item>, offset: usize, value: Self::Item) {
        set_memory(ptr.as_ptr(), offset, value)
    }

    fn clone_ptr(&self, ptr: NonNull<Self::Item>, len: usize) -> NonNull<Self::Item> {
        // malloc on device
        let cloned_ptr = cuda_malloc(len).unwrap();
        cuda_copy(
            cloned_ptr,
            ptr.as_ptr(),
            len,
            ZenuCudaMemCopyKind::HostToHost,
        )
        .unwrap();
        NonNull::new(cloned_ptr).unwrap()
    }

    fn drop(&self, ptr: *const Self::Item, _len: usize) {
        cuda_free(ptr as *mut Self::Item).unwrap();
    }

    fn offset_ptr(&self, ptr: NonNull<Self::Item>, offset: usize) -> NonNull<Self::Item> {
        NonNull::new(unsafe { ptr.as_ptr().add(offset) }).unwrap()
    }
}

#[derive(Debug)]
pub struct OwnedMem<T: Num, A: MemoryAccessor<Item = T>> {
    ptr: NonNull<T>,
    offset: usize,
    length: usize,
    accessor: A,
}

#[derive(Debug)]
pub struct ViewMem<'a, T: Num, A: MemoryAccessor<Item = T>> {
    ptr: &'a OwnedMem<T, A>,
    offset: usize,
}

#[derive(Debug)]
pub struct ViewMutMem<'a, T: Num, A: MemoryAccessor<Item = T>> {
    ptr: &'a mut OwnedMem<T, A>,
    offset: usize,
}

impl<T: Num, A: MemoryAccessor<Item = T>> Memory for OwnedMem<T, A> {
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

impl<T: Num, A: MemoryAccessor<Item = T>> Clone for OwnedMem<T, A> {
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

impl<T, A> ToViewMemory for OwnedMem<T, A>
where
    T: Num,
    A: MemoryAccessor<Item = T>,
{
    fn to_view(&self, offset: usize) -> ViewMem<T, A> {
        ViewMem { ptr: self, offset }
    }
}

impl<T, A> ToViewMutMemory for OwnedMem<T, A>
where
    T: Num,
    A: MemoryAccessor<Item = T>,
{
    fn to_view_mut(&mut self, offset: usize) -> ViewMutMem<'_, T, A> {
        ViewMutMem { ptr: self, offset }
    }
}

impl<'a, T, A> ToViewMutMemory for ViewMutMem<'a, T, A>
where
    T: Num,
    A: MemoryAccessor<Item = T>,
{
    fn to_view_mut(&mut self, offset: usize) -> ViewMutMem<'_, T, A> {
        let offset = self.get_offset() + offset;
        ViewMutMem {
            ptr: self.ptr,
            offset,
        }
    }
}

impl<T: Num, A: MemoryAccessor<Item = T>> ToOwnedMemory for OwnedMem<T, A> {
    type Owned = OwnedMem<T, A>;

    fn to_owned_memory(&self) -> Self::Owned {
        self.clone()
    }
}

impl<T: Num, A: MemoryAccessor<Item = T>> Owned for OwnedMem<T, A> {
    fn from_vec(vec: Vec<Self::Item>) -> Self {
        let ptr = unsafe { NonNull::new_unchecked(vec.as_ptr() as *mut T) };
        let length = vec.len();
        std::mem::forget(vec);
        Self {
            ptr,
            offset: 0,
            length,
            accessor: Cpu::new(),
        }
    }
}

impl<T: Num, A: MemoryAccessor<Item = T>> Drop for OwnedMem<T, A> {
    fn drop(&mut self) {
        self.accessor.drop(self.ptr.as_ptr(), self.len());
    }
}

impl<T: Num, A: MemoryAccessor<Item = T>> Clone for ViewMem<'_, T, A> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            offset: self.offset,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct OwnedMemSer<T> {
    data: Vec<T>,
    offset: usize,
    length: usize,
}

impl<T: Num + Serialize, A: MemoryAccessor<Item = T>> Serialize for OwnedMem<T, A> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let data: Vec<T> = unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr().add(self.offset), self.length).to_vec()
        };
        let ser = OwnedMemSer {
            data,
            offset: self.offset,
            length: self.length,
        };
        ser.serialize(serializer)
    }
}

impl<'de, T: Num, A: MemoryAccessor<Item = T>> Deserialize<'de> for OwnedMem<T, A>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let ser = OwnedMemSer::<T>::deserialize(deserializer)?;
        let mut data = ser.data;
        let ptr = NonNull::new(data.as_mut_ptr()).ok_or_else(|| {
            serde::de::Error::custom("Failed to create NonNull pointer from deserialized data")
        })?;
        let owned_mem = OwnedMem {
            ptr,
            offset: ser.offset,
            length: ser.length,
            accessor: Cpu::new(),
        };
        std::mem::forget(data);
        Ok(owned_mem)
    }
}
#[derive(Serialize, Deserialize)]
struct ViewMemSer<T> {
    data: Vec<T>,
    offset: usize,
}

impl<'a, T: Num + Serialize, A: MemoryAccessor<Item = A>> Serialize for ViewMem<'a, T, A> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let data: Vec<T> = unsafe {
            std::slice::from_raw_parts(
                self.ptr.ptr.as_ptr().add(self.ptr.offset + self.offset),
                self.ptr.length - self.offset,
            )
            .to_vec()
        };
        let ser = ViewMemSer {
            data,
            offset: self.offset,
        };
        ser.serialize(serializer)
    }
}

impl<'de, 'a, T: Num, A: MemoryAccessor<Item = T>> Deserialize<'de> for ViewMem<'a, T, A>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let ser = ViewMemSer::<T>::deserialize(deserializer)?;
        let owned_mem = OwnedMem {
            ptr: NonNull::dangling(),
            offset: 0,
            length: ser.data.len(),
            accessor: Cpu::new(),
        };
        let view_mem = ViewMem {
            ptr: unsafe { &*Box::into_raw(Box::new(owned_mem)) },
            offset: ser.offset,
        };
        Ok(view_mem)
    }
}

#[derive(Serialize, Deserialize)]
struct ViewMutMemSer<T> {
    data: Vec<T>,
    offset: usize,
}

impl<'a, T: Num + Serialize, A: MemoryAccessor<Item = T>> Serialize for ViewMutMem<'a, T, A> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let data: Vec<T> = unsafe {
            std::slice::from_raw_parts(
                self.ptr.ptr.as_ptr().add(self.ptr.offset + self.offset),
                self.ptr.length - self.offset,
            )
            .to_vec()
        };
        let ser = ViewMutMemSer {
            data,
            offset: self.offset,
        };
        ser.serialize(serializer)
    }
}

impl<'de, 'a, T: Num, A: MemoryAccessor<Item = T>> Deserialize<'de> for ViewMutMem<'a, T, A>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let ser = ViewMutMemSer::<T>::deserialize(deserializer)?;
        let owned_mem = OwnedMem {
            ptr: NonNull::dangling(),
            offset: 0,
            length: ser.data.len(),
            accessor: Cpu::new(),
        };
        let view_mem = ViewMutMem {
            ptr: unsafe { &mut *Box::into_raw(Box::new(owned_mem)) },
            offset: ser.offset,
        };
        Ok(view_mem)
    }
}
macro_rules! impl_cpu_memory_to_view {
    ($impl_ty: ty) => {
        impl<'a, T: Num, A: MemoryAccessor<Item = T>> Memory for $impl_ty {
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

        impl<'a, T, A: MemoryAccessor<Item = T>> ToViewMemory for $impl_ty
        where
            T: Num,
        {
            // type View<'b> = ViewMem<'b, T> where Self: 'b;

            fn to_view(&self, offset: usize) -> ViewMem<'_, T, A> {
                ViewMem {
                    ptr: self.ptr,
                    offset: self.get_offset() + offset,
                }
            }
        }

        impl<'a, T, A: MemoryAccessor<Item = T>> ToOwnedMemory for $impl_ty
        where
            T: Num,
        {
            type Owned = OwnedMem<T, A>;

            fn to_owned_memory(&self) -> Self::Owned {
                let mut memory = self.ptr.clone();
                memory.set_offset(self.offset + self.ptr.get_offset());
                memory
            }
        }
    };
}
impl_cpu_memory_to_view!(ViewMem<'a, T, A>);
impl_cpu_memory_to_view!(ViewMutMem<'a, T, A>);

impl<'a, T: Num, A: MemoryAccessor<Item = T>> View for ViewMem<'a, T, A> {}
impl<'a, T: Num, A: MemoryAccessor<Item = T>> ViewMut for ViewMutMem<'a, T, A> {
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
