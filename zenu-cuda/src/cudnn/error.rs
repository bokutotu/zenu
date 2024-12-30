use zenu_cudnn_sys::{cudnnGetErrorString, cudnnStatus_t};

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ZenuCudnnError {
    NotInitialized = 1001,
    LicenseError = 1005,
    RuntimeInProgress = 1006,
    RuntimeFpOverflow = 1007,
    BadParam = 2000,
    NotSupported = 3000,
    InternalError = 4000,
    ExecutionFailed = 5000,
    InvalidValue = 2001,
    Other = 9999,
}

impl From<cudnnStatus_t> for ZenuCudnnError {
    fn from(status: cudnnStatus_t) -> Self {
        match status {
            cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED => ZenuCudnnError::NotInitialized,
            cudnnStatus_t::CUDNN_STATUS_LICENSE_ERROR => ZenuCudnnError::LicenseError,
            cudnnStatus_t::CUDNN_STATUS_RUNTIME_IN_PROGRESS => ZenuCudnnError::RuntimeInProgress,
            cudnnStatus_t::CUDNN_STATUS_RUNTIME_FP_OVERFLOW => ZenuCudnnError::RuntimeFpOverflow,
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => ZenuCudnnError::BadParam,
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => ZenuCudnnError::NotSupported,
            cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR => ZenuCudnnError::InternalError,
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => ZenuCudnnError::ExecutionFailed,
            cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE => ZenuCudnnError::InvalidValue,
            _ => unreachable!(),
        }
    }
}

impl From<ZenuCudnnError> for cudnnStatus_t {
    fn from(error: ZenuCudnnError) -> Self {
        match error {
            ZenuCudnnError::NotInitialized => cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
            ZenuCudnnError::LicenseError => cudnnStatus_t::CUDNN_STATUS_LICENSE_ERROR,
            ZenuCudnnError::RuntimeInProgress => cudnnStatus_t::CUDNN_STATUS_RUNTIME_IN_PROGRESS,
            ZenuCudnnError::RuntimeFpOverflow => cudnnStatus_t::CUDNN_STATUS_RUNTIME_FP_OVERFLOW,
            ZenuCudnnError::BadParam => cudnnStatus_t::CUDNN_STATUS_BAD_PARAM,
            ZenuCudnnError::NotSupported => cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ZenuCudnnError::InternalError => cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR,
            ZenuCudnnError::ExecutionFailed => cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED,
            ZenuCudnnError::InvalidValue => cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE,
            ZenuCudnnError::Other => unimplemented!(),
        }
    }
}

impl std::fmt::Debug for ZenuCudnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let error: cudnnStatus_t = (*self).into();
        let error_char_ptr = unsafe { cudnnGetErrorString(error) };
        let error_str = unsafe { std::ffi::CStr::from_ptr(error_char_ptr) };
        write!(f, "{}", error_str.to_str().unwrap())
    }
}

impl std::fmt::Display for ZenuCudnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let error: cudnnStatus_t = (*self).into();
        let error_char_ptr = unsafe { cudnnGetErrorString(error) };
        let error_str = unsafe { std::ffi::CStr::from_ptr(error_char_ptr) };
        write!(f, "{}", error_str.to_str().unwrap())
    }
}
