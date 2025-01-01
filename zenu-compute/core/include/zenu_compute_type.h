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
} ZenuStatus;

#ifdef __cplusplus
}
#endif
