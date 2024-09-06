use zenu_cudnn_sys::{cudnnGetErrorString, cudnnStatus_t};

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ZenuCudnnError {
    NotInitialized = 1001,
    SublibraryVersionMismatch = 1002,
    SerializationVersionMismatch = 1003,
    Deprecated = 1004,
    LicenseError = 1005,
    RuntimeInProgress = 1006,
    RuntimeFpOverflow = 1007,
    BadParam = 2000,
    BadParamNullPointer = 2002,
    BadParamMisalignedPointer = 2003,
    BadParamNotFinalized = 2004,
    BadParamOutOfBound = 2005,
    BadParamSizeInsufficient = 2006,
    BadParamStreamMismatch = 2007,
    BadParamShapeMismatch = 2008,
    BadParamDuplicatedEntries = 2009,
    BadParamAttributeType = 2010,
    NotSupported = 3000,
    NotSupportedGraphPattern = 3001,
    NotSupportedShape = 3002,
    NotSupportedDataType = 3003,
    NotSupportedLayout = 3004,
    NotSupportedIncompatibleCUDADriver = 3005,
    NotSupportedIncompatibleCUDART = 3006,
    NotSupportedArchMismatch = 3007,
    NotSupportedRuntimePrerequisiteMissing = 3008,
    NotSupportedSublibraryUnavailable = 3009,
    NotSupportedSharedMemoryInsufficient = 3010,
    NotSupportedPadding = 3011,
    NotSupportedBadLaunchParam = 3012,
    InternalError = 4000,
    InternalErrorCompilationFailed = 4001,
    InternalErrorUnexpectedValue = 4002,
    InternalErrorHostAllocationFailed = 4003,
    InternalErrorDeviceAllocationFailed = 4004,
    InternalErrorBadLaunchParam = 4005,
    InternalErrorTextureCreationFailed = 4006,
    ExecutionFailed = 5000,
    ExecutionFailedCudaDriver = 5001,
    ExecutionFailedCublas = 5002,
    ExecutionFailedCudart = 5003,
    ExecutionFailedCurand = 5004,
    InvalidValue = 2001,
    Other = 9999,
}

impl From<cudnnStatus_t> for ZenuCudnnError {
    #[allow(clippy::too_many_lines)]
    fn from(status: cudnnStatus_t) -> Self {
        match status {
            cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED => ZenuCudnnError::NotInitialized,
            cudnnStatus_t::CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH => {
                ZenuCudnnError::SublibraryVersionMismatch
            }
            cudnnStatus_t::CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH => {
                ZenuCudnnError::SerializationVersionMismatch
            }
            cudnnStatus_t::CUDNN_STATUS_DEPRECATED => ZenuCudnnError::Deprecated,
            cudnnStatus_t::CUDNN_STATUS_LICENSE_ERROR => ZenuCudnnError::LicenseError,
            cudnnStatus_t::CUDNN_STATUS_RUNTIME_IN_PROGRESS => ZenuCudnnError::RuntimeInProgress,
            cudnnStatus_t::CUDNN_STATUS_RUNTIME_FP_OVERFLOW => ZenuCudnnError::RuntimeFpOverflow,
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => ZenuCudnnError::BadParam,
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_NULL_POINTER => {
                ZenuCudnnError::BadParamNullPointer
            }
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER => {
                ZenuCudnnError::BadParamMisalignedPointer
            }
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED => {
                ZenuCudnnError::BadParamNotFinalized
            }
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND => {
                ZenuCudnnError::BadParamOutOfBound
            }
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT => {
                ZenuCudnnError::BadParamSizeInsufficient
            }
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH => {
                ZenuCudnnError::BadParamStreamMismatch
            }
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH => {
                ZenuCudnnError::BadParamShapeMismatch
            }
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES => {
                ZenuCudnnError::BadParamDuplicatedEntries
            }
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE => {
                ZenuCudnnError::BadParamAttributeType
            }
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => ZenuCudnnError::NotSupported,
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN => {
                ZenuCudnnError::NotSupportedGraphPattern
            }
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_SHAPE => ZenuCudnnError::NotSupportedShape,
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE => {
                ZenuCudnnError::NotSupportedDataType
            }
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_LAYOUT => ZenuCudnnError::NotSupportedLayout,
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDA_DRIVER => {
                ZenuCudnnError::NotSupportedIncompatibleCUDADriver
            }
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDART => {
                ZenuCudnnError::NotSupportedIncompatibleCUDART
            }
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH => {
                ZenuCudnnError::NotSupportedArchMismatch
            }
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING => {
                ZenuCudnnError::NotSupportedRuntimePrerequisiteMissing
            }
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE => {
                ZenuCudnnError::NotSupportedSublibraryUnavailable
            }
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT => {
                ZenuCudnnError::NotSupportedSharedMemoryInsufficient
            }
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_PADDING => {
                ZenuCudnnError::NotSupportedPadding
            }
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_BAD_LAUNCH_PARAM => {
                ZenuCudnnError::NotSupportedBadLaunchParam
            }
            cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR => ZenuCudnnError::InternalError,
            cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED => {
                ZenuCudnnError::InternalErrorCompilationFailed
            }
            cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_UNEXPECTED_VALUE => {
                ZenuCudnnError::InternalErrorUnexpectedValue
            }
            cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED => {
                ZenuCudnnError::InternalErrorHostAllocationFailed
            }
            cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED => {
                ZenuCudnnError::InternalErrorDeviceAllocationFailed
            }
            cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_BAD_LAUNCH_PARAM => {
                ZenuCudnnError::InternalErrorBadLaunchParam
            }
            cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED => {
                ZenuCudnnError::InternalErrorTextureCreationFailed
            }
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => ZenuCudnnError::ExecutionFailed,
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER => {
                ZenuCudnnError::ExecutionFailedCudaDriver
            }
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED_CUBLAS => {
                ZenuCudnnError::ExecutionFailedCublas
            }
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED_CUDART => {
                ZenuCudnnError::ExecutionFailedCudart
            }
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED_CURAND => {
                ZenuCudnnError::ExecutionFailedCurand
            }
            cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE => ZenuCudnnError::InvalidValue,
            _ => unreachable!(),
        }
    }
}

impl From<ZenuCudnnError> for cudnnStatus_t {
    #[allow(clippy::too_many_lines)]
    fn from(error: ZenuCudnnError) -> Self {
        match error {
            ZenuCudnnError::NotInitialized => cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
            ZenuCudnnError::SublibraryVersionMismatch => {
                cudnnStatus_t::CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
            }
            ZenuCudnnError::SerializationVersionMismatch => {
                cudnnStatus_t::CUDNN_STATUS_SERIALIZATION_VERSION_MISMATCH
            }
            ZenuCudnnError::Deprecated => cudnnStatus_t::CUDNN_STATUS_DEPRECATED,
            ZenuCudnnError::LicenseError => cudnnStatus_t::CUDNN_STATUS_LICENSE_ERROR,
            ZenuCudnnError::RuntimeInProgress => cudnnStatus_t::CUDNN_STATUS_RUNTIME_IN_PROGRESS,
            ZenuCudnnError::RuntimeFpOverflow => cudnnStatus_t::CUDNN_STATUS_RUNTIME_FP_OVERFLOW,
            ZenuCudnnError::BadParam => cudnnStatus_t::CUDNN_STATUS_BAD_PARAM,
            ZenuCudnnError::BadParamNullPointer => {
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_NULL_POINTER
            }
            ZenuCudnnError::BadParamMisalignedPointer => {
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_MISALIGNED_POINTER
            }
            ZenuCudnnError::BadParamNotFinalized => {
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_NOT_FINALIZED
            }
            ZenuCudnnError::BadParamOutOfBound => {
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_OUT_OF_BOUND
            }
            ZenuCudnnError::BadParamSizeInsufficient => {
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT
            }
            ZenuCudnnError::BadParamStreamMismatch => {
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH
            }
            ZenuCudnnError::BadParamShapeMismatch => {
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH
            }
            ZenuCudnnError::BadParamDuplicatedEntries => {
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_DUPLICATED_ENTRIES
            }
            ZenuCudnnError::BadParamAttributeType => {
                cudnnStatus_t::CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE
            }
            ZenuCudnnError::NotSupported => cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ZenuCudnnError::NotSupportedGraphPattern => {
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN
            }
            ZenuCudnnError::NotSupportedShape => cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_SHAPE,
            ZenuCudnnError::NotSupportedDataType => {
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE
            }
            ZenuCudnnError::NotSupportedLayout => cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_LAYOUT,
            ZenuCudnnError::NotSupportedIncompatibleCUDADriver => {
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDA_DRIVER
            }
            ZenuCudnnError::NotSupportedIncompatibleCUDART => {
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_INCOMPATIBLE_CUDART
            }
            ZenuCudnnError::NotSupportedArchMismatch => {
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH
            }
            ZenuCudnnError::NotSupportedRuntimePrerequisiteMissing => {
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_RUNTIME_PREREQUISITE_MISSING
            }
            ZenuCudnnError::NotSupportedSublibraryUnavailable => {
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE
            }
            ZenuCudnnError::NotSupportedSharedMemoryInsufficient => {
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_SHARED_MEMORY_INSUFFICIENT
            }
            ZenuCudnnError::NotSupportedPadding => {
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_PADDING
            }
            ZenuCudnnError::NotSupportedBadLaunchParam => {
                cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED_BAD_LAUNCH_PARAM
            }
            ZenuCudnnError::InternalError => cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR,
            ZenuCudnnError::InternalErrorCompilationFailed => {
                cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED
            }
            ZenuCudnnError::InternalErrorUnexpectedValue => {
                cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_UNEXPECTED_VALUE
            }
            ZenuCudnnError::InternalErrorHostAllocationFailed => {
                cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED
            }
            ZenuCudnnError::InternalErrorDeviceAllocationFailed => {
                cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED
            }
            ZenuCudnnError::InternalErrorBadLaunchParam => {
                cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_BAD_LAUNCH_PARAM
            }
            ZenuCudnnError::InternalErrorTextureCreationFailed => {
                cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR_TEXTURE_CREATION_FAILED
            }
            ZenuCudnnError::ExecutionFailed => cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED,
            ZenuCudnnError::ExecutionFailedCudaDriver => {
                cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED_CUDA_DRIVER
            }
            ZenuCudnnError::ExecutionFailedCublas => {
                cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED_CUBLAS
            }
            ZenuCudnnError::ExecutionFailedCudart => {
                cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED_CUDART
            }
            ZenuCudnnError::ExecutionFailedCurand => {
                cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED_CURAND
            }
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
