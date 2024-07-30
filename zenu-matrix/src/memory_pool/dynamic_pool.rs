use std::{
    collections::{BTreeMap, HashMap},
    ops::Bound::{Included, Unbounded},
    sync::{Arc, Mutex},
};

use crate::device::DeviceBase;

use super::{dynamic_buffer::DynBuffer, MemPoolError};

#[derive(Default)]
pub struct DynMemPool<D: DeviceBase> {
    used_buffers: HashMap<*mut u8, Arc<Mutex<DynBuffer<D>>>>,
    unused_buffers: BTreeMap<usize, Vec<Arc<Mutex<DynBuffer<D>>>>>,
}

impl<D: DeviceBase> DynMemPool<D> {
    pub fn try_alloc(&mut self, bytes: usize) -> Result<*mut u8, MemPoolError> {
        match self.smallest_unused_bytes_over_request(bytes) {
            Some(smallest_unused_bytes_over_request) => {
                let buffers = self
                    .unused_buffers
                    .get_mut(&smallest_unused_bytes_over_request)
                    .unwrap();
                let buffer = buffers.pop().unwrap();
                let ptr = buffer.lock().unwrap().start_ptr();
                self.used_buffers.insert(ptr, buffer);
                if buffers.is_empty() {
                    self.unused_buffers
                        .remove(&smallest_unused_bytes_over_request);
                }
                Ok(ptr)
            }
            None => {
                let buffer = Arc::new(Mutex::new(DynBuffer::new(bytes)?));
                let ptr = buffer.lock().unwrap().start_ptr();
                self.used_buffers.insert(ptr, buffer);
                Ok(ptr)
            }
        }
    }

    pub fn try_free(&mut self, ptr: *mut u8) -> Result<(), MemPoolError> {
        let buffer = self
            .used_buffers
            .remove(&ptr)
            .ok_or(MemPoolError::DynMemPoolFreeError)?;
        let bytes = buffer.lock().unwrap().bytes();
        self.unused_buffers.entry(bytes).or_default().push(buffer);
        Ok(())
    }

    pub fn smallest_unused_bytes_over_request(&self, bytes: usize) -> Option<usize> {
        self.unused_buffers
            .range((Included(&bytes), Unbounded))
            .next()
            .map(|(unused_bytes, _)| *unused_bytes)
    }

    pub fn contains(&self, ptr: *mut u8) -> bool {
        self.used_buffers.contains_key(&ptr)
    }
}
