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
    fn try_alloc(&mut self, bytes: usize) -> Result<*mut u8, ()> {
        match self.unused_buffers.range_mut((Included(bytes), Unbounded)) {
            Some(buffer) => {}
            None => {}
        }
    }
}
