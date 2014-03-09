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
#include "sobolmoro.h"
#include "sobolmoro_kernel.h"

void sobolmoro(int n_vectors, int n_dimensions, unsigned int *d_directions, float *d_output)
{
    const int threadsperblock = 64;

    // Set up the execution configuration
    dim3 dimGrid;
    dim3 dimBlock;

    int            device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    // This implementation of the generator outputs all the draws for
    // one dimension in a contiguous region of memory, followed by the
    // next dimension and so on.
    // Therefore all threads within a block will be processing different
    // vectors from the same dimension. As a result we want the total
    // number of blocks to be a multiple of the number of dimensions.
    dimGrid.y = n_dimensions;

    // If the number of dimensions is large then we will set the number
    // of blocks to equal the number of dimensions (i.e. dimGrid.x = 1)
    // but if the number of dimensions is small (e.g. less than four per
    // multiprocessor) then we'll partition the vectors across blocks
    // (as well as threads).
    if (n_dimensions < (4 * prop.multiProcessorCount))
    {
        dimGrid.x = 4 * prop.multiProcessorCount;
    }
    else
    {
        dimGrid.x = 1;
    }

    // Cap the dimGrid.x if the number of vectors is small
    if (dimGrid.x > (unsigned int)(n_vectors / threadsperblock))
    {
        dimGrid.x = (n_vectors + threadsperblock - 1) / threadsperblock;
    }

    // Round up to a power of two, required for the algorithm so that
    // stride is a power of two.
    unsigned int targetDimGridX = dimGrid.x;

    for (dimGrid.x = 1 ; dimGrid.x < targetDimGridX ; dimGrid.x *= 2);

    // Fix the number of threads
    dimBlock.x = threadsperblock;

    // Execute GPU kernel
    sobolmoro_kernel<<<dimGrid, dimBlock>>>(n_vectors, n_dimensions, d_directions, d_output);
}