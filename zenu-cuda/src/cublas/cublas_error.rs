#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZenuCublasError {
    CublasStatusSuccess,
    CublasStatusNotInitialized,
    CublasStatusAllocFailed,
    CublasStatusInvalidValue,
    CublasStatusArchMismatch,
    CublasStatusMappingError,
    CublasStatusExecutionFailed,
    CublasStatusInternalError,
    CublasStatusNotSupported,
    CublasStatusLicenseError,
}

impl From<u32> for ZenuCublasError {
    fn from(value: u32) -> Self {
        match value {
            0 => ZenuCublasError::CublasStatusSuccess,
            1 => ZenuCublasError::CublasStatusNotInitialized,
            3 => ZenuCublasError::CublasStatusAllocFailed,
            7 => ZenuCublasError::CublasStatusInvalidValue,
            8 => ZenuCublasError::CublasStatusArchMismatch,
            11 => ZenuCublasError::CublasStatusMappingError,
            13 => ZenuCublasError::CublasStatusExecutionFailed,
            14 => ZenuCublasError::CublasStatusInternalError,
            15 => ZenuCublasError::CublasStatusNotSupported,
            16 => ZenuCublasError::CublasStatusLicenseError,
            _ => panic!("Invalid cublas error code: {}", value),
        }
    }
}

impl From<ZenuCublasError> for u32 {
    fn from(value: ZenuCublasError) -> Self {
        match value {
            ZenuCublasError::CublasStatusSuccess => 0,
            ZenuCublasError::CublasStatusNotInitialized => 1,
            ZenuCublasError::CublasStatusAllocFailed => 3,
            ZenuCublasError::CublasStatusInvalidValue => 7,
            ZenuCublasError::CublasStatusArchMismatch => 8,
            ZenuCublasError::CublasStatusMappingError => 11,
            ZenuCublasError::CublasStatusExecutionFailed => 13,
            ZenuCublasError::CublasStatusInternalError => 14,
            ZenuCublasError::CublasStatusNotSupported => 15,
            ZenuCublasError::CublasStatusLicenseError => 16,
        }
    }
}
