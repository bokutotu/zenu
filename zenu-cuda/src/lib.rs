use std::ptr::NonNull;

use zenu_cudnn_sys::{cudaError, cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyKind};

#[derive(Debug, Copy, Clone)]
pub enum ZenuCudaError {
    CudaSuccess,
    CudaErrorInvalidValue,
    CudaErrorMemoryAllocation,
    CudaErrorInitializationError,
    CudaErrorCudartUnloading,
    CudaErrorProfilerDisabled,
    CudaErrorProfilerNotInitialized,
    CudaErrorProfilerAlreadyStarted,
    CudaErrorProfilerAlreadyStopped,
    CudaErrorInvalidConfiguration,
    CudaErrorInvalidPitchValue,
    CudaErrorInvalidSymbol,
    CudaErrorInvalidHostPointer,
    CudaErrorInvalidDevicePointer,
    CudaErrorInvalidTexture,
    CudaErrorInvalidTextureBinding,
    CudaErrorInvalidChannelDescriptor,
    CudaErrorInvalidMemcpyDirection,
    CudaErrorAddressOfConstant,
    CudaErrorTextureFetchFailed,
    CudaErrorTextureNotBound,
    CudaErrorSynchronizationError,
    CudaErrorInvalidFilterSetting,
    CudaErrorInvalidNormSetting,
    CudaErrorMixedDeviceExecution,
    CudaErrorNotYetImplemented = 31,
    CudaErrorMemoryValueTooLarge,
    CudaErrorStubLibrary,
    CudaErrorInsufficientDriver,
    CudaErrorCallRequiresNewerDriver,
    CudaErrorInvalidSurface,
    CudaErrorDuplicateVariableName,
    CudaErrorDuplicateTextureName,
    CudaErrorDuplicateSurfaceName,
    CudaErrorDevicesUnavailable,
    CudaErrorIncompatibleDriverContext,
    CudaErrorMissingConfiguration,
    CudaErrorPriorLaunchFailure,
    CudaErrorLaunchMaxDepthExceeded,
    CudaErrorLaunchFileScopedTex,
    CudaErrorLaunchFileScopedSurf,
    CudaErrorSyncDepthExceeded,
    CudaErrorLaunchPendingCountExceeded,
    CudaErrorInvalidDeviceFunction,
    CudaErrorNoDevice,
    CudaErrorInvalidDevice,
    CudaErrorDeviceNotLicensed,
    CudaErrorSoftwareValidityNotEstablished,
    CudaErrorStartupFailure,
    CudaErrorInvalidKernelImage,
    CudaErrorDeviceUninitialized,
    CudaErrorMapBufferObjectFailed,
    CudaErrorUnmapBufferObjectFailed,
    CudaErrorArrayIsMapped,
    CudaErrorAlreadyMapped,
    CudaErrorNoKernelImageForDevice,
    CudaErrorAlreadyAcquired,
    CudaErrorNotMapped,
    CudaErrorNotMappedAsArray,
    CudaErrorNotMappedAsPointer,
    CudaErrorECCUncorrectable,
    CudaErrorUnsupportedLimit,
    CudaErrorDeviceAlreadyInUse,
    CudaErrorPeerAccessUnsupported,
    CudaErrorInvalidPtx,
    CudaErrorInvalidGraphicsContext,
    CudaErrorNvlinkUncorrectable,
    CudaErrorJitCompilerNotFound,
    CudaErrorUnsupportedPtxVersion,
    CudaErrorJitCompilationDisabled,
    CudaErrorUnsupportedExecAffinity,
    CudaErrorUnsupportedDevSideSync,
    CudaErrorInvalidSource,
    CudaErrorFileNotFound,
    CudaErrorSharedObjectSymbolNotFound,
    CudaErrorSharedObjectInitFailed,
    CudaErrorOperatingSystem,
    CudaErrorInvalidResourceHandle,
    CudaErrorIllegalState,
    CudaErrorLossyQuery,
    CudaErrorSymbolNotFound,
    CudaErrorNotReady,
    CudaErrorIllegalAddress,
    CudaErrorLaunchOutOfResources,
    CudaErrorLaunchTimeout,
    CudaErrorLaunchIncompatibleTexturing,
    CudaErrorPeerAccessAlreadyEnabled,
    CudaErrorPeerAccessNotEnabled,
    CudaErrorSetOnActiveProcess,
    CudaErrorContextIsDestroyed,
    CudaErrorAssert,
    CudaErrorTooManyPeers,
    CudaErrorHostMemoryAlreadyRegistered,
    CudaErrorHostMemoryNotRegistered,
    CudaErrorHardwareStackError,
    CudaErrorIllegalInstruction,
    CudaErrorMisalignedAddress,
    CudaErrorInvalidAddressSpace,
    CudaErrorInvalidPc,
    CudaErrorLaunchFailure,
    CudaErrorCooperativeLaunchTooLarge,
    CudaErrorNotPermitted,
    CudaErrorNotSupported,
    CudaErrorSystemNotReady,
    CudaErrorSystemDriverMismatch,
    CudaErrorCompatNotSupportedOnDevice,
    CudaErrorMpsConnectionFailed,
    CudaErrorMpsRpcFailure,
    CudaErrorMpsServerNotReady,
    CudaErrorMpsMaxClientsReached,
    CudaErrorMpsMaxConnectionsReached,
    CudaErrorMpsClientTerminated,
    CudaErrorCdpNotSupported,
    CudaErrorCdpVersionMismatch,
    CudaErrorStreamCaptureUnsupported,
    CudaErrorStreamCaptureInvalidated,
    CudaErrorStreamCaptureMerge,
    CudaErrorStreamCaptureUnmatched,
    CudaErrorStreamCaptureUnjoined,
    CudaErrorStreamCaptureIsolation,
    CudaErrorStreamCaptureImplicit,
    CudaErrorCapturedEvent,
    CudaErrorStreamCaptureWrongThread,
    CudaErrorTimeout,
    CudaErrorGraphExecUpdateFailure,
    CudaErrorExternalDevice,
    CudaErrorInvalidClusterSize,
    CudaErrorUnknown,
    CudaErrorApiFailureBase,
}

impl From<u32> for ZenuCudaError {
    fn from(error: u32) -> Self {
        match error {
            0 => ZenuCudaError::CudaSuccess,
            1 => ZenuCudaError::CudaErrorInvalidValue,
            2 => ZenuCudaError::CudaErrorMemoryAllocation,
            3 => ZenuCudaError::CudaErrorInitializationError,
            4 => ZenuCudaError::CudaErrorCudartUnloading,
            5 => ZenuCudaError::CudaErrorProfilerDisabled,
            6 => ZenuCudaError::CudaErrorProfilerNotInitialized,
            7 => ZenuCudaError::CudaErrorProfilerAlreadyStarted,
            8 => ZenuCudaError::CudaErrorProfilerAlreadyStopped,
            9 => ZenuCudaError::CudaErrorInvalidConfiguration,
            12 => ZenuCudaError::CudaErrorInvalidPitchValue,
            13 => ZenuCudaError::CudaErrorInvalidSymbol,
            16 => ZenuCudaError::CudaErrorInvalidHostPointer,
            17 => ZenuCudaError::CudaErrorInvalidDevicePointer,
            18 => ZenuCudaError::CudaErrorInvalidTexture,
            19 => ZenuCudaError::CudaErrorInvalidTextureBinding,
            20 => ZenuCudaError::CudaErrorInvalidChannelDescriptor,
            21 => ZenuCudaError::CudaErrorInvalidMemcpyDirection,
            22 => ZenuCudaError::CudaErrorAddressOfConstant,
            23 => ZenuCudaError::CudaErrorTextureFetchFailed,
            24 => ZenuCudaError::CudaErrorTextureNotBound,
            25 => ZenuCudaError::CudaErrorSynchronizationError,
            26 => ZenuCudaError::CudaErrorInvalidFilterSetting,
            27 => ZenuCudaError::CudaErrorInvalidNormSetting,
            28 => ZenuCudaError::CudaErrorMixedDeviceExecution,
            31 => ZenuCudaError::CudaErrorNotYetImplemented,
            32 => ZenuCudaError::CudaErrorMemoryValueTooLarge,
            34 => ZenuCudaError::CudaErrorStubLibrary,
            35 => ZenuCudaError::CudaErrorInsufficientDriver,
            36 => ZenuCudaError::CudaErrorCallRequiresNewerDriver,
            37 => ZenuCudaError::CudaErrorInvalidSurface,
            43 => ZenuCudaError::CudaErrorDuplicateVariableName,
            44 => ZenuCudaError::CudaErrorDuplicateTextureName,
            45 => ZenuCudaError::CudaErrorDuplicateSurfaceName,
            46 => ZenuCudaError::CudaErrorDevicesUnavailable,
            49 => ZenuCudaError::CudaErrorIncompatibleDriverContext,
            52 => ZenuCudaError::CudaErrorMissingConfiguration,
            53 => ZenuCudaError::CudaErrorPriorLaunchFailure,
            65 => ZenuCudaError::CudaErrorLaunchMaxDepthExceeded,
            66 => ZenuCudaError::CudaErrorLaunchFileScopedTex,
            67 => ZenuCudaError::CudaErrorLaunchFileScopedSurf,
            68 => ZenuCudaError::CudaErrorSyncDepthExceeded,
            69 => ZenuCudaError::CudaErrorLaunchPendingCountExceeded,
            98 => ZenuCudaError::CudaErrorInvalidDeviceFunction,
            100 => ZenuCudaError::CudaErrorNoDevice,
            101 => ZenuCudaError::CudaErrorInvalidDevice,
            102 => ZenuCudaError::CudaErrorDeviceNotLicensed,
            103 => ZenuCudaError::CudaErrorSoftwareValidityNotEstablished,
            127 => ZenuCudaError::CudaErrorStartupFailure,
            200 => ZenuCudaError::CudaErrorInvalidKernelImage,
            201 => ZenuCudaError::CudaErrorDeviceUninitialized,
            205 => ZenuCudaError::CudaErrorMapBufferObjectFailed,
            206 => ZenuCudaError::CudaErrorUnmapBufferObjectFailed,
            207 => ZenuCudaError::CudaErrorArrayIsMapped,
            208 => ZenuCudaError::CudaErrorAlreadyMapped,
            209 => ZenuCudaError::CudaErrorNoKernelImageForDevice,
            210 => ZenuCudaError::CudaErrorAlreadyAcquired,
            211 => ZenuCudaError::CudaErrorNotMapped,
            212 => ZenuCudaError::CudaErrorNotMappedAsArray,
            213 => ZenuCudaError::CudaErrorNotMappedAsPointer,
            214 => ZenuCudaError::CudaErrorECCUncorrectable,
            215 => ZenuCudaError::CudaErrorUnsupportedLimit,
            216 => ZenuCudaError::CudaErrorDeviceAlreadyInUse,
            217 => ZenuCudaError::CudaErrorPeerAccessUnsupported,
            218 => ZenuCudaError::CudaErrorInvalidPtx,
            219 => ZenuCudaError::CudaErrorInvalidGraphicsContext,
            220 => ZenuCudaError::CudaErrorNvlinkUncorrectable,
            221 => ZenuCudaError::CudaErrorJitCompilerNotFound,
            222 => ZenuCudaError::CudaErrorUnsupportedPtxVersion,
            223 => ZenuCudaError::CudaErrorJitCompilationDisabled,
            224 => ZenuCudaError::CudaErrorUnsupportedExecAffinity,
            225 => ZenuCudaError::CudaErrorUnsupportedDevSideSync,
            300 => ZenuCudaError::CudaErrorInvalidSource,
            301 => ZenuCudaError::CudaErrorFileNotFound,
            302 => ZenuCudaError::CudaErrorSharedObjectSymbolNotFound,
            303 => ZenuCudaError::CudaErrorSharedObjectInitFailed,
            304 => ZenuCudaError::CudaErrorOperatingSystem,
            400 => ZenuCudaError::CudaErrorInvalidResourceHandle,
            401 => ZenuCudaError::CudaErrorIllegalState,
            402 => ZenuCudaError::CudaErrorLossyQuery,
            500 => ZenuCudaError::CudaErrorSymbolNotFound,
            600 => ZenuCudaError::CudaErrorNotReady,
            700 => ZenuCudaError::CudaErrorIllegalAddress,
            701 => ZenuCudaError::CudaErrorLaunchOutOfResources,
            702 => ZenuCudaError::CudaErrorLaunchTimeout,
            703 => ZenuCudaError::CudaErrorLaunchIncompatibleTexturing,
            704 => ZenuCudaError::CudaErrorPeerAccessAlreadyEnabled,
            705 => ZenuCudaError::CudaErrorPeerAccessNotEnabled,
            708 => ZenuCudaError::CudaErrorSetOnActiveProcess,
            709 => ZenuCudaError::CudaErrorContextIsDestroyed,
            710 => ZenuCudaError::CudaErrorAssert,
            711 => ZenuCudaError::CudaErrorTooManyPeers,
            712 => ZenuCudaError::CudaErrorHostMemoryAlreadyRegistered,
            713 => ZenuCudaError::CudaErrorHostMemoryNotRegistered,
            714 => ZenuCudaError::CudaErrorHardwareStackError,
            715 => ZenuCudaError::CudaErrorIllegalInstruction,
            716 => ZenuCudaError::CudaErrorMisalignedAddress,
            717 => ZenuCudaError::CudaErrorInvalidAddressSpace,
            718 => ZenuCudaError::CudaErrorInvalidPc,
            719 => ZenuCudaError::CudaErrorLaunchFailure,
            720 => ZenuCudaError::CudaErrorCooperativeLaunchTooLarge,
            800 => ZenuCudaError::CudaErrorNotPermitted,
            801 => ZenuCudaError::CudaErrorNotSupported,
            802 => ZenuCudaError::CudaErrorSystemNotReady,
            803 => ZenuCudaError::CudaErrorSystemDriverMismatch,
            804 => ZenuCudaError::CudaErrorCompatNotSupportedOnDevice,
            805 => ZenuCudaError::CudaErrorMpsConnectionFailed,
            806 => ZenuCudaError::CudaErrorMpsRpcFailure,
            807 => ZenuCudaError::CudaErrorMpsServerNotReady,
            808 => ZenuCudaError::CudaErrorMpsMaxClientsReached,
            809 => ZenuCudaError::CudaErrorMpsMaxConnectionsReached,
            810 => ZenuCudaError::CudaErrorMpsClientTerminated,
            811 => ZenuCudaError::CudaErrorCdpNotSupported,
            812 => ZenuCudaError::CudaErrorCdpVersionMismatch,
            900 => ZenuCudaError::CudaErrorStreamCaptureUnsupported,
            901 => ZenuCudaError::CudaErrorStreamCaptureInvalidated,
            902 => ZenuCudaError::CudaErrorStreamCaptureMerge,
            903 => ZenuCudaError::CudaErrorStreamCaptureUnmatched,
            904 => ZenuCudaError::CudaErrorStreamCaptureUnjoined,
            905 => ZenuCudaError::CudaErrorStreamCaptureIsolation,
            906 => ZenuCudaError::CudaErrorStreamCaptureImplicit,
            907 => ZenuCudaError::CudaErrorCapturedEvent,
            908 => ZenuCudaError::CudaErrorStreamCaptureWrongThread,
            909 => ZenuCudaError::CudaErrorTimeout,
            910 => ZenuCudaError::CudaErrorGraphExecUpdateFailure,
            911 => ZenuCudaError::CudaErrorExternalDevice,
            912 => ZenuCudaError::CudaErrorInvalidClusterSize,
            999 => ZenuCudaError::CudaErrorUnknown,
            10000 => ZenuCudaError::CudaErrorApiFailureBase,
            _ => ZenuCudaError::CudaErrorUnknown,
        }
    }
}

pub fn cuda_malloc<T>(size: usize) -> Result<NonNull<T>, ZenuCudaError> {
    let mut ptr = std::ptr::null_mut();
    let size = size * std::mem::size_of::<T>();
    let err = unsafe { cudaMalloc(&mut ptr as *mut *mut T as *mut *mut std::ffi::c_void, size) }
        as cudaError as u32;
    let err = ZenuCudaError::from(err);
    match err {
        ZenuCudaError::CudaSuccess => Ok(unsafe { NonNull::new_unchecked(ptr) }),
        _ => Err(err),
    }
}

pub fn cuda_free<T>(ptr: NonNull<T>) -> Result<(), ZenuCudaError> {
    let err: ZenuCudaError =
        (unsafe { cudaFree(ptr.as_ptr() as *mut std::ffi::c_void) } as u32).into();
    match err {
        ZenuCudaError::CudaSuccess => Ok(()),
        _ => Err(err),
    }
}

pub enum ZenuCudaMemCopyKind {
    HostToHost,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    Default,
}

impl From<ZenuCudaMemCopyKind> for cudaMemcpyKind {
    fn from(value: ZenuCudaMemCopyKind) -> Self {
        match value {
            ZenuCudaMemCopyKind::HostToHost => cudaMemcpyKind::cudaMemcpyHostToHost,
            ZenuCudaMemCopyKind::HostToDevice => cudaMemcpyKind::cudaMemcpyHostToDevice,
            ZenuCudaMemCopyKind::DeviceToHost => cudaMemcpyKind::cudaMemcpyDeviceToHost,
            ZenuCudaMemCopyKind::DeviceToDevice => cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ZenuCudaMemCopyKind::Default => cudaMemcpyKind::cudaMemcpyDefault,
        }
    }
}

pub fn cuda_copy<T>(
    dst: NonNull<T>,
    src: NonNull<T>,
    size: usize,
    kind: ZenuCudaMemCopyKind,
) -> Result<(), ZenuCudaError> {
    let size = size * std::mem::size_of::<T>();
    let err = unsafe {
        cudaMemcpy(
            dst.as_ptr() as *mut std::ffi::c_void,
            src.as_ptr() as *mut std::ffi::c_void,
            size,
            cudaMemcpyKind::from(kind),
        )
    } as u32;
    let err = ZenuCudaError::from(err);
    match err {
        ZenuCudaError::CudaSuccess => Ok(()),
        _ => Err(err),
    }
}

#[cfg(test)]
mod cuda {
    use std::ptr::NonNull;

    use crate::{cuda_copy, cuda_malloc};

    #[test]
    fn cpu_to_gpu_to_cpu() {
        let mut a = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut b = vec![0.0f32; 4];

        let a_ptr = cuda_malloc::<f32>(4).unwrap();
        cuda_copy(
            a_ptr,
            unsafe { NonNull::new_unchecked(a.as_mut_ptr()) },
            4,
            crate::ZenuCudaMemCopyKind::HostToDevice,
        )
        .unwrap();

        cuda_copy(
            unsafe { NonNull::new_unchecked(b.as_mut_ptr()) },
            a_ptr,
            4,
            crate::ZenuCudaMemCopyKind::DeviceToHost,
        )
        .unwrap();

        assert_eq!(a, b);
    }
}
