use std::{ptr::NonNull, sync::Arc};

use ruml_matrix_traits::{
    memory::{Memory, OwnedMemory, ViewMemory},
    num::Num,
};

pub struct CpuOwnedMemory<T: Num> {
    buffer: NonNull<T>,
    len: usize,
}

impl<T: Num> Clone for CpuOwnedMemory<T> {
    fn clone(&self) -> Self {
        let v = self.as_slice().to_vec().clone();
        Self::from_vec(v)
    }
}

impl<T: Num> CpuOwnedMemory<T> {
    pub fn new(size: usize) -> Self {
        let mut v = vec![T::default(); size];
        let buffer = NonNull::new(v.as_mut_ptr()).unwrap();
        Self { buffer, len: size }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.buffer.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.as_ptr(), self.len) }
    }

    pub fn from_vec(vec: Vec<T>) -> Self {
        let mut vec = vec;
        let len = vec.len();
        let buffer = NonNull::new(vec.as_mut_ptr()).unwrap();
        std::mem::forget(vec); // Vec<T> がドロップされないようにする
        Self { buffer, len }
    }
}

impl<T: Num> Memory for CpuOwnedMemory<T> {
    type Item = T;

    fn as_ptr(&self) -> *const Self::Item {
        self.buffer.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Item {
        unsafe { self.buffer.as_mut() }
    }

    fn from_vec(vec: Vec<Self::Item>) -> Self {
        Self::from_vec(vec)
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
        let buffer = self.buffer;
        let len = self.len;
        let reference = Arc::new(Self { buffer, len });
        CpuViewMemory::new(reference, offset)
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

    fn from_vec(vec: Vec<Self::Item>) -> Self {
        let data: CpuOwnedMemory<T> = CpuOwnedMemory::from_vec(vec);
        Self::new(Arc::new(data), 0)
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

impl<T: Num> Drop for CpuOwnedMemory<T> {
    fn drop(&mut self) {
        // panic!("CpuOwnedMemory is not allowed to drop");
        unsafe {
            let _ = Vec::from_raw_parts(self.buffer.as_ptr(), self.len, self.len);
        }
    }
}

#[test]
fn test_cpu_owned_memory() {
    let v = vec![1., 2., 3., 4., 5.];
    let m = CpuOwnedMemory::from_vec(v);
    assert_eq!(m.as_slice(), &[1., 2., 3., 4., 5.]);
    assert_eq!(m.ptr_add(0), &1.);
}

#[test]
fn test_cpu_owned_to_view() {
    let v = vec![1., 2., 3., 4., 5.];
    let m = CpuOwnedMemory::from_vec(v);
    println!("here");
    let v = m.to_view(1);
    // let v_0 = unsafe { *v.as_ptr() };
    // assert_eq!(v_0, 2.);
}
