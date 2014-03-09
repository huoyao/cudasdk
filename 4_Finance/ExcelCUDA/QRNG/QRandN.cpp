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

#include "QRandN.h"

#include <cuda_runtime.h>
#include <vector>

#include "sobol.h"
#include "sobol_primitives.h"
#include "sobolmoro.h"

namespace QRand
{
    inline int success(cudaError_t result)
    {
        return result == cudaSuccess;
    }

    QRandN::QRandN(unsigned long nVectors, unsigned long nDimensions) :
        m_nVectors(nVectors),
        m_nDimensions(nDimensions)
    {
        bool ok = true;

        if (ok && ! success(cudaMalloc((void **)&m_matrix, nVectors * nDimensions * sizeof(float))))
        {
            m_matrix = 0;
            ok = false;
        }

        if (ok)
            generateMatrix();
    }

    QRandN::~QRandN()
    {
        if (m_matrix)
        {
            cudaFree(m_matrix);
            m_matrix = 0;
        }
    }

    void QRandN::generateMatrix(void)
    {
        bool ok = true;

        if (m_matrix == 0)
            ok = false;

        // Create the Sobol direction numbers on the host
        std::vector<unsigned int> h_directions(m_nDimensions * n_directions);
        initSobolDirectionVectors(m_nDimensions, &h_directions[0]);

        // Copy the direction numbers to the device and generate the
        // transformed (Normal) quasi-random vector.
        unsigned int *d_directions = 0;

        if (ok && ! success(cudaMalloc((void **)&d_directions, m_nDimensions * n_directions * sizeof(unsigned int))))
        {
            d_directions = 0;
            ok = false;
        }

        if (ok && ! success(cudaMemcpy(d_directions, &h_directions[0], m_nDimensions * n_directions * sizeof(unsigned int), cudaMemcpyHostToDevice)))
            ok = false;

        if (ok)
            sobolmoro(m_nVectors, m_nDimensions, d_directions, m_matrix);

        // Cleanup
        if (d_directions)
        {
            cudaFree(d_directions);
            d_directions = 0;
        }
    }

    float *QRandN::getDeviceMutable(void)
    {
        return m_matrix;
    }
}
