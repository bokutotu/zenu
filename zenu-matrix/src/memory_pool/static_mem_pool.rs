use std::{
    collections::{BTreeMap, HashMap},
    ops::Bound::{Included, Unbounded},
    rc::Rc,
};

use super::static_buffer::StaticSizeBuffer;
use crate::device::DeviceBase;

#[derive(Clone)]
struct RcBuffer<D: DeviceBase, const N: usize>(Rc<StaticSizeBuffer<D, N>>);

#[derive(Default)]
struct PtrBufferMap<D: DeviceBase, const N: usize>(HashMap<*mut u8, RcBuffer<D, N>>);

impl<D: DeviceBase, const N: usize> PtrBufferMap<D, N> {
    fn insert(&mut self, buffer: RcBuffer<D, N>) {
        let ptr = buffer.0.ptr();
        self.0.insert(ptr, buffer);
    }

    fn remove(&mut self, ptr: *mut u8) -> Option<RcBuffer<D, N>> {
        self.0.remove(&ptr)
    }

    fn pop(&mut self) -> (*mut u8, RcBuffer<D, N>) {
        let (ptr, _) = { &self.0.iter().next().unwrap() };
        let ptr = **ptr;
        let buffer = self.0.remove(&ptr).unwrap();
        (ptr, buffer)
    }
}

#[derive(Default)]
struct UnusedBytesPtrBufferMap<D: DeviceBase, const N: usize>(BTreeMap<usize, PtrBufferMap<D, N>>);

impl<D: DeviceBase, const N: usize> UnusedBytesPtrBufferMap<D, N> {
    /// 要求されたメモリよりも多くの未使用メモリを持つバッファの中で最小の未使用バイト数を返す
    fn smallest_unused_bytes_over_request(&self, bytes: usize) -> Option<usize> {
        self.0
            .range((Included(&bytes), Unbounded))
            .next()
            .map(|(unused_bytes, _)| *unused_bytes)
    }

    fn get_ptr_buffer_map(&mut self, unused_bytes: usize) -> Option<PtrBufferMap<D, N>> {
        self.0.remove(&unused_bytes)
    }

    fn get_unused_bytes_ptr_buffer(&mut self, unused_bytes: usize) -> RcBuffer<D, N> {
        let mut ptr_buffer_map = self.get_ptr_buffer_map(unused_bytes).unwrap();
        let (_, next_ptr_buffer_map) = ptr_buffer_map.pop();
        if !ptr_buffer_map.0.is_empty() {
            self.0.insert(unused_bytes, ptr_buffer_map).unwrap();
        }
        next_ptr_buffer_map
    }

    fn insert(&mut self, buffer: RcBuffer<D, N>) {
        let unused_bytes = buffer.0.get_unused_bytes();
        // unused_bytesに対応するPtrBufferMapが存在しない場合、新たに作成する
        self.0.entry(unused_bytes).or_default().insert(buffer);
    }
}

#[derive(Default)]
struct StaticMemPool<D: DeviceBase, const N: usize> {
    unused_bytes_ptr_buffer_map: UnusedBytesPtrBufferMap<D, N>,
    alloced_ptr_buffer_map: HashMap<*mut u8, RcBuffer<D, N>>,
}

impl<D: DeviceBase, const N: usize> StaticMemPool<D, N> {
    fn try_alloc(&mut self, bytes: usize) -> Result<*mut u8, ()> {
        if let Some(unused_bytes) = self
            .unused_bytes_ptr_buffer_map
            .smallest_unused_bytes_over_request(bytes)
        {
            let buffer = self
                .unused_bytes_ptr_buffer_map
                .get_unused_bytes_ptr_buffer(unused_bytes);
            let ptr = buffer.0.try_alloc(bytes)?;
            self.alloced_ptr_buffer_map.insert(ptr, buffer.clone());
            Ok(ptr)
        } else {
            Err(())
        }
    }
}
