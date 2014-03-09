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

#include <stdio.h>
#include <cuda.h>

#include "ExcelCUDA_wrapper.h"
#include "ExcelCUDA_kernel.h"

inline int success(cudaError_t result)
{
    return result == cudaSuccess;
}

// Simple kernel to check the connection to the GPU
void ExcelCUDA_HelloWorld(char *result, int size)
{
    bool ok = true;
    char *d_result = 0;

    // Allocate memory on device
    if (ok && ! success(cudaMalloc((void **)&d_result, sizeof(char) * size)))
    {
        strncpy(result, cudaGetErrorString(cudaGetLastError()), size);
        ok = false;
    }

    // Call the kernel
    if (ok)
        kernel_HelloWorld<<<1, 1>>>(d_result, size);

    // Get the result from the device
    if (ok && ! success(cudaMemcpy(result, d_result, sizeof(char) * size, cudaMemcpyDeviceToHost)))
    {
        strncpy(result, cudaGetErrorString(cudaGetLastError()), size);
        ok = false;
    }

    // Cleanup
    if (d_result)
        cudaFree(d_result);
}

// Use Quasi Monte Carlo to calculate Pi
void ExcelCUDA_CalculatePiMC(float *result, unsigned long n_steps)
{
    int ok = 1;

    int *h_intResults = 0;
    int *d_intResults = 0;

    float stepsize;
    dim3 block;
    dim3 grid;

    // Determine execution configuration
    block.x = 32;
    block.y = 16;
    grid.x = (n_steps + block.x - 1) / block.x;
    grid.y = (n_steps + block.y - 1) / block.y;

    // Determine the step size
    stepsize = 1.0f / (float)(n_steps - 1);

    // Allocate memory on host
    if (ok && (h_intResults = (int *)malloc(sizeof(int) * grid.x * grid.y)) == NULL)
        ok = 0;

    // Allocate memory on device
    if (ok && ! success(cudaMalloc((void **)&d_intResults, sizeof(int) * grid.x * grid.y)))
        ok = 0;

    // Call the kernel
    if (ok)
        kernel_CalculatePiMC<<<grid, block, block.x *block.y *sizeof(int)>>>(d_intResults, stepsize, n_steps);

    // Get the result from the device
    if (ok && ! success(cudaMemcpy(h_intResults, d_intResults, sizeof(int) * grid.x * grid.y, cudaMemcpyDeviceToHost)))
        ok = 0;

    // Reduce
    int count = 0;
    for (unsigned int i = 0 ; ok && i < (grid.x *grid.y) ; i++)
        count = count + h_intResults[i];

    // Compute result
    *result = (float)(4 * (double)count / ((double)n_steps * (double)n_steps));

    // Cleanup
    if (h_intResults)
    {
        free(h_intResults);
        h_intResults = 0;
    }
    if (d_intResults)
    {
        cudaFree(d_intResults);
        d_intResults = 0;
    }
}
