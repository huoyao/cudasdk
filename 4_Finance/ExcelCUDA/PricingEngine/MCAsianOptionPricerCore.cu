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

#include "MCAsianOptionPricerCore.h"
#include "MCAsianOptionPlan.h"
#include "QRandN.h"

#include <cuda.h>

namespace MCAsianOption
{
    inline int success(cudaError_t result)
    {
        return result == cudaSuccess;
    }

    __device__ float sumreduce(float in)
    {
        extern __shared__ float sdata[];

        // Perform first level of reduction:
        // - Read from global memory
        // - Write to shared memory
        unsigned int tid = threadIdx.x;

        sdata[tid] = in;
        __syncthreads();

        // Do reduction in shared mem
        for (unsigned int s = blockDim.x / 2 ; s > 0 ; s >>= 1)
        {
            if (tid < s)
            {
                sdata[tid] += sdata[tid + s];
            }

            __syncthreads();
        }

        return sdata[0];
    }

    static __inline__ __device__ void atomicAdd(float *addr, float val)
    {
        float old = *addr;
        float assumed;

        do
        {
            assumed = old;
            old = __int_as_float(atomicCAS((unsigned int *)addr,
                                           __float_as_int(assumed),
                                           __float_as_int(val + assumed)));
        }
        while (assumed != old);
    }

    __global__ void MCAsianOptionPricerKernel(MCAsianOption::MCAsianOptionPlan::MCAsianOptionDevicePlan *plan, float *draws, float *result)
    {
        // Determine offsets and strides
        unsigned int iOption = blockIdx.x;
        unsigned int iScenario = blockIdx.y * blockDim.x + threadIdx.x;
        unsigned int sScenario = gridDim.y * blockDim.x;

        // Read the option parameters
        float spot   = plan->spot[iOption];
        float strike = plan->strike[iOption];
        float r      = plan->r[iOption];
        float sigma  = plan->sigma[iOption];
        float tenor  = plan->tenor[iOption];
        MCAsianOption::MCAsianOptionPlan::CallPut callPut = plan->callPut[iOption];

        // Calculate drift/diffusion per timestep
        float dt           = tenor / ((float)plan->nTimesteps - 1);
        float drift        = (r - 0.5f * sigma * sigma) * dt;
        float diffusion    = sigma * sqrtf(dt);

        float valueA = 0;
        float valueQ = 0;

        for (unsigned int count = 1 ; iScenario < plan->nScenarios ; iScenario += sScenario, count++)
        {
            float payoffArithmetic;
            float s = 1.0f;
            float avgArithmetic = 1.0f;

            // Offset into the random numbers
            float *draw = draws + iScenario;

            for (unsigned int t = 1 ; t < plan->nTimesteps ; t++, draw += plan->nScenarios)
            {
                s             *= expf(drift + diffusion * *draw);
                avgArithmetic += s;
            }

            // Scale to the original spot price
            s             *= spot;
            avgArithmetic = spot * avgArithmetic / (float)plan->nTimesteps;

            // Payoff
            if (callPut == MCAsianOption::MCAsianOptionPlan::Call)
            {
                payoffArithmetic = max(0.0f, avgArithmetic - strike);
            }
            else // Put
            {
                payoffArithmetic = max(0.0f, strike - avgArithmetic);
            }

            valueQ = valueQ + (count - 1) * (payoffArithmetic - valueA) * (payoffArithmetic - valueA) / count;
            valueA = valueA + (payoffArithmetic - valueA) / count;
        }

        // Reduce (average) across threads within the block
        valueA = sumreduce(valueA) / (float)blockDim.x;

        // Reduce (sum) across blocks for the same option (i.e. across block.x)
        // Discount to current time
        if (threadIdx.x == 0)
            atomicAdd(&result[iOption], valueA * expf(-r * tenor));
    }

    void MCAsianOptionPricerCore(MCAsianOptionPlan *plan, MCAsianOptionPlan::MCAsianOptionDevicePlan *d_plan, float *result)
    {
        bool ok = true;
        unsigned int nOptions = plan->getNOptions();

        // Allocate memory on device for results
        float *d_result = 0;

        if (ok && ! success(cudaMalloc((void **)&d_result, nOptions * sizeof(float))))
            ok = false;

        if (ok && ! success(cudaMemset((void *)d_result, 0, nOptions * sizeof(float))))
            ok = false;

        // Create the random numbers
        QRand::QRandN draws(plan->getNScenarios(), plan->getNTimesteps());
        float *d_draws = draws.getDeviceMutable();

        if (d_draws == 0)
            ok = false;

        // Execute the kernel
        dim3 block;
        dim3 grid;
        block.x = 128;
        grid.x = nOptions;
        grid.y = (plan->getNScenarios() + block.x - 1) / block.x;

        if (ok)
            MCAsianOptionPricerKernel<<<grid, block, block.x *sizeof(float)>>>(d_plan, d_draws, d_result);

        // Copy results back to host
        if (ok && ! success(cudaMemcpy(result, d_result, nOptions * sizeof(float), cudaMemcpyDeviceToHost)))
            ok = false;

        // Complete the final part of the reduction (convert sum to average)
        for (unsigned int i = 0 ; i < nOptions ; i++)
        {
            result[i] /= grid.y;
        }

        // Check for errors
        cudaError_t cudaresult = cudaSuccess;

        if ((cudaresult = cudaGetLastError()) != cudaSuccess)
        {
            // Fill the array with a negative value (the error code)
            for (unsigned int i = 0 ; i < nOptions ; i++)
            {
                result[i] = (float)(-abs(cudaresult));
            }
        }

        // Cleanup
        if (d_result)
        {
            cudaFree(d_result);
            d_result = 0;
        }
    }
}