use zenu_cudnn_sys::cudnnStatus_t;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZenuCudnnError {
    CudnnStatusSuccess,
    CudnnStatusNotINITIALIZED,
    CudnnStatusAllocFAILED,
    CudnnStatusBadParam,
    CudnnStatusInternalError,
    CudnnStatusInvalidValue,
    CudnnStatusArchMismatch,
    CudnnStatusMappingError,
    CudnnStatusExecutionFailed,
    CjdnnStatusNotSupported,
    CudnnStatusLicenseError,
    CudnnStatusRuntimePrerequisiteMissing,
    CudnnStatusRuntimeINProgress,
    CudnnStatusRuntimeFPOverflow,
    CudnnStatusVersionMismatch,
}

impl From<u32> for ZenuCudnnError {
    fn from(value: u32) -> Self {
        match value {
            0 => ZenuCudnnError::CudnnStatusSuccess,
            1 => ZenuCudnnError::CudnnStatusNotINITIALIZED,
            2 => ZenuCudnnError::CudnnStatusAllocFAILED,
            3 => ZenuCudnnError::CudnnStatusBadParam,
            4 => ZenuCudnnError::CudnnStatusInternalError,
            7 => ZenuCudnnError::CudnnStatusInvalidValue,
            8 => ZenuCudnnError::CudnnStatusArchMismatch,
            11 => ZenuCudnnError::CudnnStatusMappingError,
            13 => ZenuCudnnError::CudnnStatusExecutionFailed,
            14 => ZenuCudnnError::CjdnnStatusNotSupported,
            15 => ZenuCudnnError::CudnnStatusLicenseError,
            16 => ZenuCudnnError::CudnnStatusRuntimePrerequisiteMissing,
            17 => ZenuCudnnError::CudnnStatusRuntimeINProgress,
            18 => ZenuCudnnError::CudnnStatusRuntimeFPOverflow,
            19 => ZenuCudnnError::CudnnStatusVersionMismatch,
            _ => panic!("Invalid cudnn error code: {}", value),
        }
    }
}

impl From<ZenuCudnnError> for u32 {
    fn from(value: ZenuCudnnError) -> Self {
        match value {
            ZenuCudnnError::CudnnStatusSuccess => 0,
            ZenuCudnnError::CudnnStatusNotINITIALIZED => 1,
            ZenuCudnnError::CudnnStatusAllocFAILED => 2,
            ZenuCudnnError::CudnnStatusBadParam => 3,
            ZenuCudnnError::CudnnStatusInternalError => 4,
            ZenuCudnnError::CudnnStatusInvalidValue => 7,
            ZenuCudnnError::CudnnStatusArchMismatch => 8,
            ZenuCudnnError::CudnnStatusMappingError => 11,
            ZenuCudnnError::CudnnStatusExecutionFailed => 13,
            ZenuCudnnError::CjdnnStatusNotSupported => 14,
            ZenuCudnnError::CudnnStatusLicenseError => 15,
            ZenuCudnnError::CudnnStatusRuntimePrerequisiteMissing => 16,
            ZenuCudnnError::CudnnStatusRuntimeINProgress => 17,
            ZenuCudnnError::CudnnStatusRuntimeFPOverflow => 18,
            ZenuCudnnError::CudnnStatusVersionMismatch => 19,
        }
    }
}

impl From<ZenuCudnnError> for cudnnStatus_t {
    fn from(value: ZenuCudnnError) -> Self {
        match value {
            ZenuCudnnError::CudnnStatusSuccess => cudnnStatus_t::CUDNN_STATUS_SUCCESS,
            ZenuCudnnError::CudnnStatusNotINITIALIZED => {
                cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED
            }
            ZenuCudnnError::CudnnStatusAllocFAILED => cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED,
            ZenuCudnnError::CudnnStatusBadParam => cudnnStatus_t::CUDNN_STATUS_BAD_PARAM,
            ZenuCudnnError::CudnnStatusInternalError => cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR,
            ZenuCudnnError::CudnnStatusInvalidValue => cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE,
            ZenuCudnnError::CudnnStatusArchMismatch => cudnnStatus_t::CUDNN_STATUS_ARCH_MISMATCH,
            ZenuCudnnError::CudnnStatusMappingError => cudnnStatus_t::CUDNN_STATUS_MAPPING_ERROR,
            ZenuCudnnError::CudnnStatusExecutionFailed => {
                cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED
            }
            ZenuCudnnError::CjdnnStatusNotSupported => cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED,
            ZenuCudnnError::CudnnStatusLicenseError => cudnnStatus_t::CUDNN_STATUS_LICENSE_ERROR,
            ZenuCudnnError::CudnnStatusRuntimePrerequisiteMissing => {
                cudnnStatus_t::CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING
            }
            ZenuCudnnError::CudnnStatusRuntimeINProgress => {
                cudnnStatus_t::CUDNN_STATUS_RUNTIME_IN_PROGRESS
            }
            ZenuCudnnError::CudnnStatusRuntimeFPOverflow => {
                cudnnStatus_t::CUDNN_STATUS_RUNTIME_FP_OVERFLOW
            }
            ZenuCudnnError::CudnnStatusVersionMismatch => {
                cudnnStatus_t::CUDNN_STATUS_VERSION_MISMATCH
            }
        }
    }
}

impl From<cudnnStatus_t> for ZenuCudnnError {
    fn from(value: cudnnStatus_t) -> Self {
        match value {
            cudnnStatus_t::CUDNN_STATUS_SUCCESS => ZenuCudnnError::CudnnStatusSuccess,
            cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED => {
                ZenuCudnnError::CudnnStatusNotINITIALIZED
            }
            cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => ZenuCudnnError::CudnnStatusAllocFAILED,
            cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => ZenuCudnnError::CudnnStatusBadParam,
            cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR => ZenuCudnnError::CudnnStatusInternalError,
            cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE => ZenuCudnnError::CudnnStatusInvalidValue,
            cudnnStatus_t::CUDNN_STATUS_ARCH_MISMATCH => ZenuCudnnError::CudnnStatusArchMismatch,
            cudnnStatus_t::CUDNN_STATUS_MAPPING_ERROR => ZenuCudnnError::CudnnStatusMappingError,
            cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => {
                ZenuCudnnError::CudnnStatusExecutionFailed
            }
            cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => ZenuCudnnError::CjdnnStatusNotSupported,
            cudnnStatus_t::CUDNN_STATUS_LICENSE_ERROR => ZenuCudnnError::CudnnStatusLicenseError,
            cudnnStatus_t::CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING => {
                ZenuCudnnError::CudnnStatusRuntimePrerequisiteMissing
            }
            cudnnStatus_t::CUDNN_STATUS_RUNTIME_IN_PROGRESS => {
                ZenuCudnnError::CudnnStatusRuntimeINProgress
            }
            cudnnStatus_t::CUDNN_STATUS_RUNTIME_FP_OVERFLOW => {
                ZenuCudnnError::CudnnStatusRuntimeFPOverflow
            }
            cudnnStatus_t::CUDNN_STATUS_VERSION_MISMATCH => {
                ZenuCudnnError::CudnnStatusVersionMismatch
            }
            _ => panic!("Invalid cudnn error code: {:?}", value),
        }
    }
}
