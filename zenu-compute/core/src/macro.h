#pragma once

#include <iostream>

#define DEFINE_DATA_SIZE(type, data_bytes)              \
switch (type) {                                         \
case ZenuDataType::f32:                                 \
    data_bytes = sizeof(float);                         \
    break;                                              \
case ZenuDataType::f64:                                 \
    data_bytes = sizeof(double);                        \
    break;                                              \
default:                                                \
    std::cout << "Unsupported data type" << std::endl;  \
    exit(1);                                            \
}
