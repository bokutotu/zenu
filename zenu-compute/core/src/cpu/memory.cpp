#include "zenu_compute.h"
#include <cstdlib>
#include <cstring>

ZenuStatus zenu_compute_malloc_cpu(void** ptr, int num_bytes) {
    *ptr = malloc(num_bytes);
    if (*ptr == NULL) {
        return ZenuStatus::OutOfMemory;
    }
    return ZenuStatus::Success;
}

void zenu_compute_free_cpu(void* ptr) {
    free(ptr);
}

void zenu_compute_copy_cpu(void* dst, void* src, int num_bytes) {
    // TODO: Copy memory on the CPU
}

void zenu_compute_set_cpu(void* dst, int value, int num_bytes) {
    memset(dst, value, num_bytes);
}
