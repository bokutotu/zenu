#pragma once

#include "zenu_compute_type.h"
#ifdef __cplusplus
extern "C" {
#endif

#include "zenu_compute.h"

ZenuStatus zenu_compute_malloc_cpu(void** ptr, int num_bytes);

ZenuStatus zenu_compute_malloc_nvidia(void** ptr, int num_bytes);

void zenu_compute_free_cpu(void* ptr);

void zenu_compute_free_nvidia(void* ptr);

void zenu_compute_set_cpu(void* dst, void* value, int num_bytes, ZenuDataType type);

ZenuStatus zenu_compute_set_nvidia(void* dst, void* value, int num_bytes, ZenuDataType type);

ZenuStatus zenu_compute_cpu_to_nvidia(void* dst, void* src, int num_bytes);

ZenuStatus zenu_compute_nvidia_to_cpu(void* dst, void* src, int num_bytes);

#ifdef __cplusplus
}
#endif
