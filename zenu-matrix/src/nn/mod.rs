use crate::device::DeviceBase;

pub mod batch_norm;
pub mod col2im;
pub mod conv2d;
pub mod dropout;
pub mod im2col;
pub mod pool2d;

#[allow(unused)]
pub(crate) struct NNCache<D: DeviceBase> {
    pub(crate) bytes: usize,
    pub(crate) ptr: *mut u8,
    _device: std::marker::PhantomData<D>,
}

impl<D: DeviceBase> NNCache<D> {
    #[allow(unused)]
    pub(crate) fn new(bytes: usize) -> Self {
        let ptr = D::alloc(bytes).unwrap();
        if ptr.is_null() {
            panic!("Failed to allocate memory");
        }
        Self {
            bytes,
            ptr,
            _device: std::marker::PhantomData,
        }
    }
}

impl<D: DeviceBase> Drop for NNCache<D> {
    fn drop(&mut self) {
        D::drop_ptr(self.ptr);
    }
}
