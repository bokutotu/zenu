#pragma once

#ifdef __cplusplus
extern "C" {
#endif

ZenuStatus zenu_compute_normal_distribution_cpu(void* dst,
                                            int num_elements,
                                            float mean,
                                            float stddev,
                                            ZenuDataType data_type);

ZenuStatus zenu_compute_normal_distribution_nvidia(void* dst,
                                            int num_elements,
                                            float mean,
                                            float stddev,
                                            ZenuDataType data_type,
                                            int device_id);

ZenuStatus zenu_compute_uniform_distribution_cpu(void* dst,
                                             int num_elements,
                                             float low,
                                             float high,
                                             ZenuDataType data_type);

ZenuStatus zenu_compute_uniform_distribution_nvidia(void* dst,
                                             int num_elements,
                                             float low,
                                             float high,
                                             ZenuDataType data_type,
                                             int device_id);

#ifdef __cplusplus
}
#endif
