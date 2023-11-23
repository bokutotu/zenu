use std::{ptr::NonNull, sync::Arc};

use ruml_matrix_traits::{
    memory::{Memory, OwnedMemory, ViewMemory},
    num::Num,
};

#[derive(Clone)]
pub struct CpuOwnedMemory<T: Num> {
    buffer: NonNull<T>,
    len: usize,
}

impl<T: Num> CpuOwnedMemory<T> {
    pub fn new(size: usize) -> Self {
        let mut buffer = vec![T::default(); size];
        let buffer = NonNull::new(buffer.as_mut_ptr()).unwrap();
        Self { buffer, len: size }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.buffer.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.as_ptr(), self.len) }
    }

    pub fn from_vec(vec: Vec<T>) -> Self {
        let len = vec.len();
        let mut buffer = vec;
        let buffer = NonNull::new(buffer.as_mut_ptr()).unwrap();
        Self { buffer, len }
    }
}

impl<T: Num> Memory for CpuOwnedMemory<T> {
    type Item = T;

    fn as_ptr(&self) -> *const Self::Item {
        self.buffer.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Item {
        self.buffer.as_ptr()
    }
}

impl<T: Num> OwnedMemory for CpuOwnedMemory<T> {
    type View = CpuViewMemory<T>;

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        todo!();
    }

    fn allocate(size: usize) -> Self {
        Self::new(size)
    }

    fn to_view(&self, offset: usize) -> Self::View {
        CpuViewMemory::new(Arc::new(self.clone()), offset)
    }
}

#[derive(Clone)]
pub struct CpuViewMemory<T: Num> {
    reference: Arc<CpuOwnedMemory<T>>,
    offset: usize,
}

impl<T: Num> CpuViewMemory<T> {
    pub fn new(reference: Arc<CpuOwnedMemory<T>>, offset: usize) -> Self {
        Self { reference, offset }
    }
}

impl<T: Num> Memory for CpuViewMemory<T> {
    type Item = T;

    fn as_ptr(&self) -> *const Self::Item {
        unsafe { self.reference.as_ptr().add(self.offset) }
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Item {
        self.as_ptr() as *mut _
    }
}

impl<T: Num> ViewMemory for CpuViewMemory<T> {
    type Owned = CpuOwnedMemory<T>;
    fn offset(&self) -> usize {
        self.offset
    }

    fn to_owned(&self) -> CpuOwnedMemory<T> {
        let v = self.reference.as_slice().to_vec().clone();
        CpuOwnedMemory::from_vec(v)
    }
}
