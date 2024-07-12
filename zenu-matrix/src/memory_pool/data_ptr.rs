use crate::device::DeviceBase;

pub(super) struct DataPtr<D: DeviceBase> {
    pub ptr: *mut u8,
    pub bytes: usize,
    _marker: std::marker::PhantomData<D>,
}

impl<D: DeviceBase> DataPtr<D> {
    pub(super) fn new(bytes: usize) -> Result<Self, ()> {
        let ptr = D::alloc(bytes)?;
        Ok(DataPtr {
            ptr,
            bytes,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<D: DeviceBase> Drop for DataPtr<D> {
    fn drop(&mut self) {
        D::drop_ptr(self.ptr, self.bytes);
    }
}
