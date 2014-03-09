/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

    void ExcelCUDA_HelloWorld(char *result, int size);
    void ExcelCUDA_CalculatePiMC(float *result, unsigned long n_iterations);

#ifdef __cplusplus
}
#endif
