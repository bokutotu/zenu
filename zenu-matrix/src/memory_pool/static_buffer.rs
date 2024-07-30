use crate::device::DeviceBase;

use super::{data_ptr::DataPtr, MemPoolError, MIDDLE_BUFFER_SIZE};
use std::collections::BTreeMap;

pub struct StaticSizeBuffer<D: DeviceBase, const N: usize> {
    pub data: DataPtr<D>,
    // key is sttart address of used buffer
    // value is end address of used buffer
    used_buffer_range: BTreeMap<*mut u8, *mut u8>,
}

impl<D: DeviceBase, const N: usize> StaticSizeBuffer<D, N> {
    pub fn new() -> Result<Self, MemPoolError> {
        Ok(StaticSizeBuffer {
            data: DataPtr::new(N)?,
            used_buffer_range: BTreeMap::new(),
        })
    }

    fn last_ptr(&self) -> Option<*mut u8> {
        self.used_buffer_range
            .last_key_value()
            .map(|(_, value)| *value)
    }

    // 確保するメモリの始点と終点を返す
    fn start_end_ptr(&self, bytes: usize) -> (*mut u8, *mut u8) {
        let (start, end) = self
            .last_ptr()
            .map(|end| {
                let start = unsafe { end.add(MIDDLE_BUFFER_SIZE) };
                let end = unsafe { start.add(bytes) };
                (start, end)
            })
            .unwrap_or_else(|| {
                let start = self.data.ptr;
                let end = unsafe { start.add(bytes) };
                (start, end)
            });
        // 最初と最後のポインタが正しいかチェック
        assert!(start >= self.data.ptr);
        assert!(end <= unsafe { self.data.ptr.add(N) });
        (start, end)
    }

    pub fn try_alloc(&mut self, bytes: usize) -> Result<*mut u8, MemPoolError> {
        if self.get_unused_bytes() < bytes {
            return Err(MemPoolError::StaticBufferTooLargeRequestError);
        }
        let (mut start, mut end) = self.start_end_ptr(bytes);

        if start as usize % 8 != 0 {
            start = unsafe { start.sub(start as usize % 8) };
        }
        if end as usize % 8 != 0 {
            end = unsafe { end.add(8 - end as usize % 8) };
        }

        if start < self.data.ptr || end > unsafe { self.data.ptr.add(N) } {
            return Err(MemPoolError::StaticBufferPtrRangeError);
        }

        self.used_buffer_range.insert(start, end);
        Ok(start)
    }

    pub fn try_free(&mut self, ptr: *mut u8) -> Result<(), MemPoolError> {
        self.used_buffer_range
            .remove(&ptr)
            .ok_or(MemPoolError::StaticBufferFreeError)?;
        Ok(())
    }

    pub fn get_unused_bytes(&self) -> usize {
        let alloced_last_ptr = self.data.ptr.wrapping_add(N);
        match self.last_ptr() {
            None => N,
            Some(last_ptr) => {
                if (alloced_last_ptr as usize) <= (last_ptr as usize + MIDDLE_BUFFER_SIZE) {
                    return 0;
                }
                alloced_last_ptr as usize - (last_ptr as usize + MIDDLE_BUFFER_SIZE)
            }
        }
    }

    pub fn ptr(&self) -> *mut u8 {
        self.data.ptr
    }
}

impl<D: DeviceBase, const N: usize> PartialEq for StaticSizeBuffer<D, N> {
    fn eq(&self, other: &Self) -> bool {
        self.data.ptr == other.data.ptr
    }
}

impl<D: DeviceBase, const N: usize> Eq for StaticSizeBuffer<D, N> {}
