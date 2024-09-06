#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Copy, Clone)]
pub enum ZenuCudaRuntimeError {
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

impl From<u32> for ZenuCudaRuntimeError {
    #[allow(clippy::too_many_lines)]
    fn from(error: u32) -> Self {
        match error {
            0 => ZenuCudaRuntimeError::CudaSuccess,
            1 => ZenuCudaRuntimeError::CudaErrorInvalidValue,
            2 => ZenuCudaRuntimeError::CudaErrorMemoryAllocation,
            3 => ZenuCudaRuntimeError::CudaErrorInitializationError,
            4 => ZenuCudaRuntimeError::CudaErrorCudartUnloading,
            5 => ZenuCudaRuntimeError::CudaErrorProfilerDisabled,
            6 => ZenuCudaRuntimeError::CudaErrorProfilerNotInitialized,
            7 => ZenuCudaRuntimeError::CudaErrorProfilerAlreadyStarted,
            8 => ZenuCudaRuntimeError::CudaErrorProfilerAlreadyStopped,
            9 => ZenuCudaRuntimeError::CudaErrorInvalidConfiguration,
            12 => ZenuCudaRuntimeError::CudaErrorInvalidPitchValue,
            13 => ZenuCudaRuntimeError::CudaErrorInvalidSymbol,
            16 => ZenuCudaRuntimeError::CudaErrorInvalidHostPointer,
            17 => ZenuCudaRuntimeError::CudaErrorInvalidDevicePointer,
            18 => ZenuCudaRuntimeError::CudaErrorInvalidTexture,
            19 => ZenuCudaRuntimeError::CudaErrorInvalidTextureBinding,
            20 => ZenuCudaRuntimeError::CudaErrorInvalidChannelDescriptor,
            21 => ZenuCudaRuntimeError::CudaErrorInvalidMemcpyDirection,
            22 => ZenuCudaRuntimeError::CudaErrorAddressOfConstant,
            23 => ZenuCudaRuntimeError::CudaErrorTextureFetchFailed,
            24 => ZenuCudaRuntimeError::CudaErrorTextureNotBound,
            25 => ZenuCudaRuntimeError::CudaErrorSynchronizationError,
            26 => ZenuCudaRuntimeError::CudaErrorInvalidFilterSetting,
            27 => ZenuCudaRuntimeError::CudaErrorInvalidNormSetting,
            28 => ZenuCudaRuntimeError::CudaErrorMixedDeviceExecution,
            31 => ZenuCudaRuntimeError::CudaErrorNotYetImplemented,
            32 => ZenuCudaRuntimeError::CudaErrorMemoryValueTooLarge,
            34 => ZenuCudaRuntimeError::CudaErrorStubLibrary,
            35 => ZenuCudaRuntimeError::CudaErrorInsufficientDriver,
            36 => ZenuCudaRuntimeError::CudaErrorCallRequiresNewerDriver,
            37 => ZenuCudaRuntimeError::CudaErrorInvalidSurface,
            43 => ZenuCudaRuntimeError::CudaErrorDuplicateVariableName,
            44 => ZenuCudaRuntimeError::CudaErrorDuplicateTextureName,
            45 => ZenuCudaRuntimeError::CudaErrorDuplicateSurfaceName,
            46 => ZenuCudaRuntimeError::CudaErrorDevicesUnavailable,
            49 => ZenuCudaRuntimeError::CudaErrorIncompatibleDriverContext,
            52 => ZenuCudaRuntimeError::CudaErrorMissingConfiguration,
            53 => ZenuCudaRuntimeError::CudaErrorPriorLaunchFailure,
            65 => ZenuCudaRuntimeError::CudaErrorLaunchMaxDepthExceeded,
            66 => ZenuCudaRuntimeError::CudaErrorLaunchFileScopedTex,
            67 => ZenuCudaRuntimeError::CudaErrorLaunchFileScopedSurf,
            68 => ZenuCudaRuntimeError::CudaErrorSyncDepthExceeded,
            69 => ZenuCudaRuntimeError::CudaErrorLaunchPendingCountExceeded,
            98 => ZenuCudaRuntimeError::CudaErrorInvalidDeviceFunction,
            100 => ZenuCudaRuntimeError::CudaErrorNoDevice,
            101 => ZenuCudaRuntimeError::CudaErrorInvalidDevice,
            102 => ZenuCudaRuntimeError::CudaErrorDeviceNotLicensed,
            103 => ZenuCudaRuntimeError::CudaErrorSoftwareValidityNotEstablished,
            127 => ZenuCudaRuntimeError::CudaErrorStartupFailure,
            200 => ZenuCudaRuntimeError::CudaErrorInvalidKernelImage,
            201 => ZenuCudaRuntimeError::CudaErrorDeviceUninitialized,
            205 => ZenuCudaRuntimeError::CudaErrorMapBufferObjectFailed,
            206 => ZenuCudaRuntimeError::CudaErrorUnmapBufferObjectFailed,
            207 => ZenuCudaRuntimeError::CudaErrorArrayIsMapped,
            208 => ZenuCudaRuntimeError::CudaErrorAlreadyMapped,
            209 => ZenuCudaRuntimeError::CudaErrorNoKernelImageForDevice,
            210 => ZenuCudaRuntimeError::CudaErrorAlreadyAcquired,
            211 => ZenuCudaRuntimeError::CudaErrorNotMapped,
            212 => ZenuCudaRuntimeError::CudaErrorNotMappedAsArray,
            213 => ZenuCudaRuntimeError::CudaErrorNotMappedAsPointer,
            214 => ZenuCudaRuntimeError::CudaErrorECCUncorrectable,
            215 => ZenuCudaRuntimeError::CudaErrorUnsupportedLimit,
            216 => ZenuCudaRuntimeError::CudaErrorDeviceAlreadyInUse,
            217 => ZenuCudaRuntimeError::CudaErrorPeerAccessUnsupported,
            218 => ZenuCudaRuntimeError::CudaErrorInvalidPtx,
            219 => ZenuCudaRuntimeError::CudaErrorInvalidGraphicsContext,
            220 => ZenuCudaRuntimeError::CudaErrorNvlinkUncorrectable,
            221 => ZenuCudaRuntimeError::CudaErrorJitCompilerNotFound,
            222 => ZenuCudaRuntimeError::CudaErrorUnsupportedPtxVersion,
            223 => ZenuCudaRuntimeError::CudaErrorJitCompilationDisabled,
            224 => ZenuCudaRuntimeError::CudaErrorUnsupportedExecAffinity,
            225 => ZenuCudaRuntimeError::CudaErrorUnsupportedDevSideSync,
            300 => ZenuCudaRuntimeError::CudaErrorInvalidSource,
            301 => ZenuCudaRuntimeError::CudaErrorFileNotFound,
            302 => ZenuCudaRuntimeError::CudaErrorSharedObjectSymbolNotFound,
            303 => ZenuCudaRuntimeError::CudaErrorSharedObjectInitFailed,
            304 => ZenuCudaRuntimeError::CudaErrorOperatingSystem,
            400 => ZenuCudaRuntimeError::CudaErrorInvalidResourceHandle,
            401 => ZenuCudaRuntimeError::CudaErrorIllegalState,
            402 => ZenuCudaRuntimeError::CudaErrorLossyQuery,
            500 => ZenuCudaRuntimeError::CudaErrorSymbolNotFound,
            600 => ZenuCudaRuntimeError::CudaErrorNotReady,
            700 => ZenuCudaRuntimeError::CudaErrorIllegalAddress,
            701 => ZenuCudaRuntimeError::CudaErrorLaunchOutOfResources,
            702 => ZenuCudaRuntimeError::CudaErrorLaunchTimeout,
            703 => ZenuCudaRuntimeError::CudaErrorLaunchIncompatibleTexturing,
            704 => ZenuCudaRuntimeError::CudaErrorPeerAccessAlreadyEnabled,
            705 => ZenuCudaRuntimeError::CudaErrorPeerAccessNotEnabled,
            708 => ZenuCudaRuntimeError::CudaErrorSetOnActiveProcess,
            709 => ZenuCudaRuntimeError::CudaErrorContextIsDestroyed,
            710 => ZenuCudaRuntimeError::CudaErrorAssert,
            711 => ZenuCudaRuntimeError::CudaErrorTooManyPeers,
            712 => ZenuCudaRuntimeError::CudaErrorHostMemoryAlreadyRegistered,
            713 => ZenuCudaRuntimeError::CudaErrorHostMemoryNotRegistered,
            714 => ZenuCudaRuntimeError::CudaErrorHardwareStackError,
            715 => ZenuCudaRuntimeError::CudaErrorIllegalInstruction,
            716 => ZenuCudaRuntimeError::CudaErrorMisalignedAddress,
            717 => ZenuCudaRuntimeError::CudaErrorInvalidAddressSpace,
            718 => ZenuCudaRuntimeError::CudaErrorInvalidPc,
            719 => ZenuCudaRuntimeError::CudaErrorLaunchFailure,
            720 => ZenuCudaRuntimeError::CudaErrorCooperativeLaunchTooLarge,
            800 => ZenuCudaRuntimeError::CudaErrorNotPermitted,
            801 => ZenuCudaRuntimeError::CudaErrorNotSupported,
            802 => ZenuCudaRuntimeError::CudaErrorSystemNotReady,
            803 => ZenuCudaRuntimeError::CudaErrorSystemDriverMismatch,
            804 => ZenuCudaRuntimeError::CudaErrorCompatNotSupportedOnDevice,
            805 => ZenuCudaRuntimeError::CudaErrorMpsConnectionFailed,
            806 => ZenuCudaRuntimeError::CudaErrorMpsRpcFailure,
            807 => ZenuCudaRuntimeError::CudaErrorMpsServerNotReady,
            808 => ZenuCudaRuntimeError::CudaErrorMpsMaxClientsReached,
            809 => ZenuCudaRuntimeError::CudaErrorMpsMaxConnectionsReached,
            810 => ZenuCudaRuntimeError::CudaErrorMpsClientTerminated,
            811 => ZenuCudaRuntimeError::CudaErrorCdpNotSupported,
            812 => ZenuCudaRuntimeError::CudaErrorCdpVersionMismatch,
            900 => ZenuCudaRuntimeError::CudaErrorStreamCaptureUnsupported,
            901 => ZenuCudaRuntimeError::CudaErrorStreamCaptureInvalidated,
            902 => ZenuCudaRuntimeError::CudaErrorStreamCaptureMerge,
            903 => ZenuCudaRuntimeError::CudaErrorStreamCaptureUnmatched,
            904 => ZenuCudaRuntimeError::CudaErrorStreamCaptureUnjoined,
            905 => ZenuCudaRuntimeError::CudaErrorStreamCaptureIsolation,
            906 => ZenuCudaRuntimeError::CudaErrorStreamCaptureImplicit,
            907 => ZenuCudaRuntimeError::CudaErrorCapturedEvent,
            908 => ZenuCudaRuntimeError::CudaErrorStreamCaptureWrongThread,
            909 => ZenuCudaRuntimeError::CudaErrorTimeout,
            910 => ZenuCudaRuntimeError::CudaErrorGraphExecUpdateFailure,
            911 => ZenuCudaRuntimeError::CudaErrorExternalDevice,
            912 => ZenuCudaRuntimeError::CudaErrorInvalidClusterSize,
            10000 => ZenuCudaRuntimeError::CudaErrorApiFailureBase,
            _ => ZenuCudaRuntimeError::CudaErrorUnknown,
        }
    }
}

impl From<ZenuCudaRuntimeError> for u32 {
    #[allow(clippy::too_many_lines)]
    fn from(error: ZenuCudaRuntimeError) -> Self {
        match error {
            ZenuCudaRuntimeError::CudaSuccess => 0,
            ZenuCudaRuntimeError::CudaErrorInvalidValue => 1,
            ZenuCudaRuntimeError::CudaErrorMemoryAllocation => 2,
            ZenuCudaRuntimeError::CudaErrorInitializationError => 3,
            ZenuCudaRuntimeError::CudaErrorCudartUnloading => 4,
            ZenuCudaRuntimeError::CudaErrorProfilerDisabled => 5,
            ZenuCudaRuntimeError::CudaErrorProfilerNotInitialized => 6,
            ZenuCudaRuntimeError::CudaErrorProfilerAlreadyStarted => 7,
            ZenuCudaRuntimeError::CudaErrorProfilerAlreadyStopped => 8,
            ZenuCudaRuntimeError::CudaErrorInvalidConfiguration => 9,
            ZenuCudaRuntimeError::CudaErrorInvalidPitchValue => 12,
            ZenuCudaRuntimeError::CudaErrorInvalidSymbol => 13,
            ZenuCudaRuntimeError::CudaErrorInvalidHostPointer => 16,
            ZenuCudaRuntimeError::CudaErrorInvalidDevicePointer => 17,
            ZenuCudaRuntimeError::CudaErrorInvalidTexture => 18,
            ZenuCudaRuntimeError::CudaErrorInvalidTextureBinding => 19,
            ZenuCudaRuntimeError::CudaErrorInvalidChannelDescriptor => 20,
            ZenuCudaRuntimeError::CudaErrorInvalidMemcpyDirection => 21,
            ZenuCudaRuntimeError::CudaErrorAddressOfConstant => 22,
            ZenuCudaRuntimeError::CudaErrorTextureFetchFailed => 23,
            ZenuCudaRuntimeError::CudaErrorTextureNotBound => 24,
            ZenuCudaRuntimeError::CudaErrorSynchronizationError => 25,
            ZenuCudaRuntimeError::CudaErrorInvalidFilterSetting => 26,
            ZenuCudaRuntimeError::CudaErrorInvalidNormSetting => 27,
            ZenuCudaRuntimeError::CudaErrorMixedDeviceExecution => 28,
            ZenuCudaRuntimeError::CudaErrorNotYetImplemented => 31,
            ZenuCudaRuntimeError::CudaErrorMemoryValueTooLarge => 32,
            ZenuCudaRuntimeError::CudaErrorStubLibrary => 34,
            ZenuCudaRuntimeError::CudaErrorInsufficientDriver => 35,
            ZenuCudaRuntimeError::CudaErrorCallRequiresNewerDriver => 36,
            ZenuCudaRuntimeError::CudaErrorInvalidSurface => 37,
            ZenuCudaRuntimeError::CudaErrorDuplicateVariableName => 43,
            ZenuCudaRuntimeError::CudaErrorDuplicateTextureName => 44,
            ZenuCudaRuntimeError::CudaErrorDuplicateSurfaceName => 45,
            ZenuCudaRuntimeError::CudaErrorDevicesUnavailable => 46,
            ZenuCudaRuntimeError::CudaErrorIncompatibleDriverContext => 49,
            ZenuCudaRuntimeError::CudaErrorMissingConfiguration => 52,
            ZenuCudaRuntimeError::CudaErrorPriorLaunchFailure => 53,
            ZenuCudaRuntimeError::CudaErrorLaunchMaxDepthExceeded => 65,
            ZenuCudaRuntimeError::CudaErrorLaunchFileScopedTex => 66,
            ZenuCudaRuntimeError::CudaErrorLaunchFileScopedSurf => 67,
            ZenuCudaRuntimeError::CudaErrorSyncDepthExceeded => 68,
            ZenuCudaRuntimeError::CudaErrorLaunchPendingCountExceeded => 69,
            ZenuCudaRuntimeError::CudaErrorInvalidDeviceFunction => 98,
            ZenuCudaRuntimeError::CudaErrorNoDevice => 100,
            ZenuCudaRuntimeError::CudaErrorInvalidDevice => 101,
            ZenuCudaRuntimeError::CudaErrorDeviceNotLicensed => 102,
            ZenuCudaRuntimeError::CudaErrorSoftwareValidityNotEstablished => 103,
            ZenuCudaRuntimeError::CudaErrorStartupFailure => 127,
            ZenuCudaRuntimeError::CudaErrorInvalidKernelImage => 200,
            ZenuCudaRuntimeError::CudaErrorDeviceUninitialized => 201,
            ZenuCudaRuntimeError::CudaErrorMapBufferObjectFailed => 205,
            ZenuCudaRuntimeError::CudaErrorUnmapBufferObjectFailed => 206,
            ZenuCudaRuntimeError::CudaErrorArrayIsMapped => 207,
            ZenuCudaRuntimeError::CudaErrorAlreadyMapped => 208,
            ZenuCudaRuntimeError::CudaErrorNoKernelImageForDevice => 209,
            ZenuCudaRuntimeError::CudaErrorAlreadyAcquired => 210,
            ZenuCudaRuntimeError::CudaErrorNotMapped => 211,
            ZenuCudaRuntimeError::CudaErrorNotMappedAsArray => 212,
            ZenuCudaRuntimeError::CudaErrorNotMappedAsPointer => 213,
            ZenuCudaRuntimeError::CudaErrorECCUncorrectable => 214,
            ZenuCudaRuntimeError::CudaErrorUnsupportedLimit => 215,
            ZenuCudaRuntimeError::CudaErrorDeviceAlreadyInUse => 216,
            ZenuCudaRuntimeError::CudaErrorPeerAccessUnsupported => 217,
            ZenuCudaRuntimeError::CudaErrorInvalidPtx => 218,
            ZenuCudaRuntimeError::CudaErrorInvalidGraphicsContext => 219,
            ZenuCudaRuntimeError::CudaErrorNvlinkUncorrectable => 220,
            ZenuCudaRuntimeError::CudaErrorJitCompilerNotFound => 221,
            ZenuCudaRuntimeError::CudaErrorUnsupportedPtxVersion => 222,
            ZenuCudaRuntimeError::CudaErrorJitCompilationDisabled => 223,
            ZenuCudaRuntimeError::CudaErrorUnsupportedExecAffinity => 224,
            ZenuCudaRuntimeError::CudaErrorUnsupportedDevSideSync => 225,
            ZenuCudaRuntimeError::CudaErrorInvalidSource => 300,
            ZenuCudaRuntimeError::CudaErrorFileNotFound => 301,
            ZenuCudaRuntimeError::CudaErrorSharedObjectSymbolNotFound => 302,
            ZenuCudaRuntimeError::CudaErrorSharedObjectInitFailed => 303,
            ZenuCudaRuntimeError::CudaErrorOperatingSystem => 304,
            ZenuCudaRuntimeError::CudaErrorInvalidResourceHandle => 400,
            ZenuCudaRuntimeError::CudaErrorIllegalState => 401,
            ZenuCudaRuntimeError::CudaErrorLossyQuery => 402,
            ZenuCudaRuntimeError::CudaErrorSymbolNotFound => 500,
            ZenuCudaRuntimeError::CudaErrorNotReady => 600,
            ZenuCudaRuntimeError::CudaErrorIllegalAddress => 700,
            ZenuCudaRuntimeError::CudaErrorLaunchOutOfResources => 701,
            ZenuCudaRuntimeError::CudaErrorLaunchTimeout => 702,
            ZenuCudaRuntimeError::CudaErrorLaunchIncompatibleTexturing => 703,
            ZenuCudaRuntimeError::CudaErrorPeerAccessAlreadyEnabled => 704,
            ZenuCudaRuntimeError::CudaErrorPeerAccessNotEnabled => 705,
            ZenuCudaRuntimeError::CudaErrorSetOnActiveProcess => 708,
            ZenuCudaRuntimeError::CudaErrorContextIsDestroyed => 709,
            ZenuCudaRuntimeError::CudaErrorAssert => 710,
            ZenuCudaRuntimeError::CudaErrorTooManyPeers => 711,
            ZenuCudaRuntimeError::CudaErrorHostMemoryAlreadyRegistered => 712,
            ZenuCudaRuntimeError::CudaErrorHostMemoryNotRegistered => 713,
            ZenuCudaRuntimeError::CudaErrorHardwareStackError => 714,
            ZenuCudaRuntimeError::CudaErrorIllegalInstruction => 715,
            ZenuCudaRuntimeError::CudaErrorMisalignedAddress => 716,
            ZenuCudaRuntimeError::CudaErrorInvalidAddressSpace => 717,
            ZenuCudaRuntimeError::CudaErrorInvalidPc => 718,
            ZenuCudaRuntimeError::CudaErrorLaunchFailure => 719,
            ZenuCudaRuntimeError::CudaErrorCooperativeLaunchTooLarge => 720,
            ZenuCudaRuntimeError::CudaErrorNotPermitted => 800,
            ZenuCudaRuntimeError::CudaErrorNotSupported => 801,
            ZenuCudaRuntimeError::CudaErrorSystemNotReady => 802,
            ZenuCudaRuntimeError::CudaErrorSystemDriverMismatch => 803,
            ZenuCudaRuntimeError::CudaErrorCompatNotSupportedOnDevice => 804,
            ZenuCudaRuntimeError::CudaErrorMpsConnectionFailed => 805,
            ZenuCudaRuntimeError::CudaErrorMpsRpcFailure => 806,
            ZenuCudaRuntimeError::CudaErrorMpsServerNotReady => 807,
            ZenuCudaRuntimeError::CudaErrorMpsMaxClientsReached => 808,
            ZenuCudaRuntimeError::CudaErrorMpsMaxConnectionsReached => 809,
            ZenuCudaRuntimeError::CudaErrorMpsClientTerminated => 810,
            ZenuCudaRuntimeError::CudaErrorCdpNotSupported => 811,
            ZenuCudaRuntimeError::CudaErrorCdpVersionMismatch => 812,
            ZenuCudaRuntimeError::CudaErrorStreamCaptureUnsupported => 900,
            ZenuCudaRuntimeError::CudaErrorStreamCaptureInvalidated => 901,
            ZenuCudaRuntimeError::CudaErrorStreamCaptureMerge => 902,
            ZenuCudaRuntimeError::CudaErrorStreamCaptureUnmatched => 903,
            ZenuCudaRuntimeError::CudaErrorStreamCaptureUnjoined => 904,
            ZenuCudaRuntimeError::CudaErrorStreamCaptureIsolation => 905,
            ZenuCudaRuntimeError::CudaErrorStreamCaptureImplicit => 906,
            ZenuCudaRuntimeError::CudaErrorCapturedEvent => 907,
            ZenuCudaRuntimeError::CudaErrorStreamCaptureWrongThread => 908,
            ZenuCudaRuntimeError::CudaErrorTimeout => 909,
            ZenuCudaRuntimeError::CudaErrorGraphExecUpdateFailure => 910,
            ZenuCudaRuntimeError::CudaErrorExternalDevice => 911,
            ZenuCudaRuntimeError::CudaErrorInvalidClusterSize => 912,
            ZenuCudaRuntimeError::CudaErrorUnknown => 999,
            ZenuCudaRuntimeError::CudaErrorApiFailureBase => 10000,
        }
    }
}
