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

/*
 * Portions Copyright (c) 1993-2013 NVIDIA Corporation.  All rights reserved.
 * Portions Copyright (c) 2009 Mike Giles, Oxford University.  All rights reserved.
 * Portions Copyright (c) 2008 Frances Y. Kuo and Stephen Joe.  All rights reserved.
 *
 * Sobol Quasi-random Number Generator example
 *
 * Based on CUDA code submitted by Mike Giles, Oxford University, United Kingdom
 * http://people.maths.ox.ac.uk/~gilesm/
 *
 * and C code developed by Stephen Joe, University of Waikato, New Zealand
 * and Frances Kuo, University of New South Wales, Australia
 * http://web.maths.unsw.edu.au/~fkuo/sobol/
 *
 * For theoretical background see:
 *
 * P. Bratley and B.L. Fox.
 * Implementing Sobol's quasirandom sequence generator
 * http://portal.acm.org/citation.cfm?id=42288
 * ACM Trans. on Math. Software, 14(1):88-100, 1988
 *
 * S. Joe and F. Kuo.
 * Remark on algorithm 659: implementing Sobol's quasirandom sequence generator.
 * http://portal.acm.org/citation.cfm?id=641879
 * ACM Trans. on Math. Software, 29(1):49-57, 2003
 *
 */

#include <cuda.h>
#include "sobol.h"
#include "sobolmoro_kernel.h"

#define k_2powneg32 2.3283064E-10F

__device__ inline float MoroInvCND(float P)
{
    const float a1 = 2.50662823884f;
    const float a2 = -18.61500062529f;
    const float a3 = 41.39119773534f;
    const float a4 = -25.44106049637f;
    const float b1 = -8.4735109309f;
    const float b2 = 23.08336743743f;
    const float b3 = -21.06224101826f;
    const float b4 = 3.13082909833f;
    const float c1 = 0.337475482272615f;
    const float c2 = 0.976169019091719f;
    const float c3 = 0.160797971491821f;
    const float c4 = 2.76438810333863E-02f;
    const float c5 = 3.8405729373609E-03f;
    const float c6 = 3.951896511919E-04f;
    const float c7 = 3.21767881768E-05f;
    const float c8 = 2.888167364E-07f;
    const float c9 = 3.960315187E-07f;
    float y;
    float z;

    if (P <= 0 || P >= 1.0f)
        return __int_as_float(0x7FFFFFFF);

    y = P - 0.5f;

    if (fabsf(y) < 0.42f)
    {
        z = y * y;
        z = y * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + 1.0f);
    }
    else
    {
        if (y > 0)
            z = __logf(-__logf(1.0f - P));
        else
            z = __logf(-__logf(P));

        z = c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9)))))));

        if (y < 0)
            z = -z;
    }

    return z;
}

__global__ void sobolmoro_kernel(unsigned n_vectors, unsigned n_dimensions, unsigned *d_directions, float *d_output)
{
    __shared__ unsigned int v[n_directions];

    // Offset into the correct dimension as specified by the
    // block y coordinate
    d_directions = d_directions + n_directions * blockIdx.y;
    d_output = d_output +  n_vectors * blockIdx.y;

    // Copy the direction numbers for this dimension into shared
    // memory - there are only 32 direction numbers so only the
    // first 32 (n_directions) threads need participate.
    if (threadIdx.x < n_directions)
    {
        v[threadIdx.x] = d_directions[threadIdx.x];
    }

    __syncthreads();

    // Set initial index (i.e. which vector this thread is
    // computing first) and stride (i.e. step to the next vector
    // for this thread)
    int i0     = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    // Get the gray code of the index
    // c.f. Numerical Recipes in C, chapter 20
    // http://www.nrbook.com/a/bookcpdf/c20-2.pdf
    unsigned int g = (i0 + 1) ^ ((i0 + 1) >> 1);

    // Initialisation for first point x[i0]
    // In the Bratley and Fox paper this is equation (*), where
    // we are computing the value for x[n] without knowing the
    // value of x[n-1].
    unsigned int X = 0;
    unsigned int mask;

    for (unsigned int k = 0 ; k < __ffs(stride) - 1 ; k++)
    {
        // We want X ^= g_k * v[k], where g_k is one or zero.
        // We do this by setting a mask with all bits equal to
        // g_k. In reality we keep shifting g so that g_k is the
        // LSB of g. This way we avoid multiplication.
        mask = - (g & 1);
        X ^= mask & v[k];
        g = g >> 1;
    }

    if (i0 < n_vectors)
    {
        d_output[i0] = MoroInvCND((float)X * k_2powneg32);
    }

    // Now do rest of points, using the stride
    // Here we want to generate x[i] from x[i-stride] where we
    // don't have any of the x in between, therefore we have to
    // revisit the equation (**), this is easiest with an example
    // so assume stride is 16.
    // From x[n] to x[n+16] there will be:
    //   8 changes in the first bit
    //   4 changes in the second bit
    //   2 changes in the third bit
    //   1 change in the fourth
    //   1 change in one of the remaining bits
    //
    // What this means is that in the equation:
    //   x[n+1] = x[n] ^ v[p]
    //   x[n+2] = x[n+1] ^ v[q] = x[n] ^ v[p] ^ v[q]
    //   ...
    // We will apply xor with v[1] eight times, v[2] four times,
    // v[3] twice, v[4] once and one other direction number once.
    // Since two xors cancel out, we can skip even applications
    // and just apply xor with v[4] (i.e. log2(16)) and with
    // the current applicable direction number.
    // Note that all these indices count from 1, so we need to
    // subtract 1 from them all to account for C arrays counting
    // from zero.
    unsigned int v_log2stridem1 = v[__ffs(stride) - 2];
    unsigned int v_stridemask = stride - 1;

    for (unsigned int i = i0 + stride ; i < n_vectors ; i += stride)
    {
        // x[i] = x[i-stride] ^ v[b] ^ v[c]
        //  where b is log2(stride) minus 1 for C array indexing
        //  where c is the index of the rightmost zero bit in i,
        //  not including the bottom log2(stride) bits, minus 1
        //  for C array indexing
        // In the Bratley and Fox paper this is equation (**)
        X ^= v_log2stridem1 ^ v[__ffs(~((i - stride) | v_stridemask)) - 1];
        d_output[i] = MoroInvCND((float)X * k_2powneg32);
    }
}
