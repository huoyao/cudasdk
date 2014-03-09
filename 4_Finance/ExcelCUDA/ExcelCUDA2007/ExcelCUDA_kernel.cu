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

#include <cuda.h>
#include "ExcelCUDA_kernel.h"

// Simple hello world kernel
__global__ void kernel_HelloWorld(char *result, int num)
{
    int i = 0;
    char p_HelloCUDA[] = "Hello CUDA!";

    for (i = 0 ; i < num ; i++)
    {
        result[i] = p_HelloCUDA[i];
    }
}

// Simple grid computation for Pi
__global__ void kernel_CalculatePiMC(int *d_intResults, float stepsize, unsigned long n_steps)
{
    extern __shared__ int sdata[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned long pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long pos_y = blockIdx.y * blockDim.y + threadIdx.y;
    float x = (float)pos_x * stepsize;
    float y = (float)pos_y * stepsize;

    if (pos_x < n_steps && pos_y < n_steps && (x * x + y * y) < 1)
    {
        sdata[tid] = 1;
    }
    else
    {
        sdata[tid] = 0;
    }

    __syncthreads();

    for (unsigned int s = (blockDim.x * blockDim.y) / 2 ; s > 0 ; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        d_intResults[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
    }
}
