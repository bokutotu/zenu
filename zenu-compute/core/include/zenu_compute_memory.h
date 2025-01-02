#pragma once

#ifdef __cplusplus
extern "C" {
#endif

ZenuStatus zenu_compute_malloc_cpu(void** ptr, int num_bytes);

ZenuStatus zenu_compute_malloc_nvidia(void** ptr, int num_bytes);

void zenu_compute_free_cpu(void* ptr);

void zenu_compute_free_nvidia(void* ptr);

void zenu_compute_copy_cpu(void* dst, void* src, int num_bytes);

void zenu_compute_copy_nvidia(void* dst, void* src, int num_bytes);

void zenu_compute_set_cpu(void* dst, int value, int num_bytes);

void zenu_compute_set_nvidia(void* dst, int value, int num_bytes);

#ifdef __cplusplus
}
#endif
