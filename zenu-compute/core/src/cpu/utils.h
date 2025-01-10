#pragma once

#include "zenu_compute.h"

/*=========================================================
 * ヘルパー: 引数チェック
 *========================================================*/
static inline ZenuStatus
check_common_args(const void* dst, size_t n, ZenuDataType dt)
{
    if (!dst)                 return InvalidArgument;
    if (dt != f32 && dt != f64) return InvalidArgument;
    /* n=0 は何もしなくてもよいので Success を返す */
    return (n == 0) ? Success : Success;
}
