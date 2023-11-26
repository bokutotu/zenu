use std::ptr::NonNull;

use ruml_matrix_traits::{
    memory::{Memory, OwnedMemory, ViewMemory},
    num::Num,
};

pub struct CpuOwnedMemory<T: Num> {
    buffer: NonNull<T>,
    len: usize,
    offset: usize,
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
        Self {
            buffer,
            len: size,
            offset: 0,
        }
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
        Self {
            buffer,
            len,
            offset: 0,
        }
    }

    pub fn from_vec_with_offset(vec: Vec<T>, offset: usize) -> Self {
        let mut vec = vec;
        let len = vec.len();
        let buffer = NonNull::new(vec.as_mut_ptr()).unwrap();
        std::mem::forget(vec); // Vec<T> がドロップされないようにする
        Self {
            buffer,
            len,
            offset,
        }
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

    fn get_offset(&self) -> usize {
        self.offset
    }
}

impl<'a, T: Num + 'a> OwnedMemory<'a> for CpuOwnedMemory<T> {
    type View = CpuViewMemory<'a, T>;

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        todo!();
    }

    fn allocate(size: usize) -> Self {
        Self::new(size)
    }

    fn to_view(&'a self, offset: usize) -> Self::View {
        CpuViewMemory::new(self, offset)
    }

    fn from_vec(vec: Vec<Self::Item>) -> Self {
        Self::from_vec(vec)
    }
}

#[derive(Clone)]
pub struct CpuViewMemory<'a, T: Num> {
    reference: &'a CpuOwnedMemory<T>,
    offset: usize,
}

impl<'a, T: Num> CpuViewMemory<'a, T> {
    pub fn new(reference: &'a CpuOwnedMemory<T>, offset: usize) -> Self {
        Self { reference, offset }
    }

    pub fn reference(&self) -> &'a CpuOwnedMemory<T> {
        self.reference
    }
}

impl<'a, T: Num> Memory for CpuViewMemory<'a, T> {
    type Item = T;

    fn as_ptr(&self) -> *const Self::Item {
        unsafe { self.reference.as_ptr().add(self.offset) }
    }

    fn as_mut_ptr(&mut self) -> *mut Self::Item {
        self.as_ptr() as *mut _
    }

    fn get_offset(&self) -> usize {
        self.offset
    }
}

impl<'a, 'b, T: Num + 'b> ViewMemory<'b> for CpuViewMemory<'a, T> {
    type Owned = CpuOwnedMemory<T>;
    fn offset(&self) -> usize {
        self.offset
    }

    fn to_owned(&self) -> CpuOwnedMemory<T> {
        let v = self.reference.as_slice().to_vec().clone();
        let offset = self.offset;
        CpuOwnedMemory::from_vec_with_offset(v, offset)
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
    let v = m.to_view(1);
    let v_0 = unsafe { *v.as_ptr() };
    assert_eq!(v_0, 2.);
}
