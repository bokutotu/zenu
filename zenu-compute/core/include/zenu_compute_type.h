#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    f32,
    f64,
} ZenuDataType;

typedef enum {
    Success,
    OutOfMemory,
    InvalidArgument,
    DeviceError,
} ZenuStatus;

typedef enum {
    Transpose,
    NoTranspose,
} ZenuTranspose;

#ifdef __cplusplus
}
#endif
