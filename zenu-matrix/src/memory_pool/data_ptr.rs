use crate::device::DeviceBase;

use super::MemPoolError;

pub(super) struct DataPtr<D: DeviceBase> {
    pub ptr: *mut u8,
    pub bytes: usize,
    _marker: std::marker::PhantomData<D>,
}

impl<D: DeviceBase> DataPtr<D> {
    pub(super) fn new(bytes: usize) -> Result<Self, MemPoolError> {
        let ptr = D::raw_alloc(bytes).map_err(|_| MemPoolError::DataPtrError)? as *mut u8;
        Ok(DataPtr {
            ptr,
            bytes,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<D: DeviceBase> Drop for DataPtr<D> {
    fn drop(&mut self) {
        D::raw_drop_ptr(self.ptr);
    }
}
