use crate::device::DeviceBase;

use super::data_ptr::DataPtr;

pub struct DynBuffer<D: DeviceBase> {
    data_ptr: DataPtr<D>,
    is_used: bool,
}

impl<D: DeviceBase> DynBuffer<D> {
    pub fn new(bytes: usize) -> Result<Self, ()> {
        Ok(DynBuffer {
            data_ptr: DataPtr::new(bytes)?,
            is_used: false,
        })
    }

    pub fn is_used(&self) -> bool {
        self.is_used
    }

    pub fn set_used(&mut self, is_used: bool) {
        self.is_used = is_used;
    }
}
