use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    ops::Bound::{Included, Unbounded},
    rc::Rc,
};

use crate::device::DeviceBase;

use super::dynamic_buffer::DynBuffer;

#[derive(Default)]
pub struct DynMemPool<D: DeviceBase> {
    used_buffers: HashMap<*mut u8, Rc<RefCell<DynBuffer<D>>>>,
    unused_buffers: BTreeMap<usize, Vec<Rc<RefCell<DynBuffer<D>>>>>,
}

impl<D: DeviceBase> DynMemPool<D> {
    pub fn try_alloc(&mut self, bytes: usize) -> Result<*mut u8, ()> {
        // match self
        //     .unused_buffers
        //     .range_mut((Included(bytes), Unbounded))
        //     .next()
        // {
        match self.unused_buffers.get_mut(&bytes) {
            Some(buffers) => {
                let buffer = buffers.pop().unwrap();
                let ptr = buffer.borrow().start_ptr();
                self.used_buffers.insert(ptr, buffer);
                Ok(ptr)
            }
            None => {
                let buffer = Rc::new(RefCell::new(DynBuffer::new(bytes)?));
                let ptr = buffer.borrow().start_ptr();
                self.used_buffers.insert(ptr, buffer);
                Ok(ptr)
            }
        }
    }

    pub fn try_free(&mut self, ptr: *mut u8) -> Result<(), ()> {
        let buffer = self.used_buffers.remove(&ptr).ok_or(())?;
        let bytes = buffer.borrow().bytes();
        self.unused_buffers
            .entry(bytes)
            .or_insert_with(Vec::new)
            .push(buffer);
        Ok(())
    }

    fn smallest_unused_bytes_over_request(&self, bytes: usize) -> Option<usize> {
        self.unused_buffers
            .range((Included(&bytes), Unbounded))
            .next()
            .map(|(unused_bytes, _)| *unused_bytes)
    }
}
