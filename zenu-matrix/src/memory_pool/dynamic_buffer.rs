use crate::device::DeviceBase;

use super::data_ptr::DataPtr;

pub struct DynBuffer<D: DeviceBase> {
    data_ptr: DataPtr<D>,
}

impl<D: DeviceBase> DynBuffer<D> {
    pub fn new(bytes: usize) -> Result<Self, ()> {
        Ok(DynBuffer {
            data_ptr: DataPtr::new(bytes)?,
        })
    }

    pub fn start_ptr(&self) -> *mut u8 {
        self.data_ptr.ptr
    }

    pub fn bytes(&self) -> usize {
        self.data_ptr.bytes
    }
}
