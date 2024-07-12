use std::ptr::NonNull;

pub(super) struct DataPtr<D: DeviceBase> {
    pub ptr: NonNull<u8>,
    pub bytes: usize,
}

impl<D: DeviceBase> DataPtr<D> {
    pub(super) fn new(bytes: usize) -> Self {
        let ptr = D::alloc(bytes);
        DataPtr { ptr, bytes }
    }
}

impl<D: DeviceBase> Drop for DataPtr<D> {
    fn drop(&mut self) {
        D::free(self.ptr, self.bytes);
    }
}
