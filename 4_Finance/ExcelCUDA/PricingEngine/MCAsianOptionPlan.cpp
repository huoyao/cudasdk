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

#include "MCAsianOptionPlan.h"

#include <cuda_runtime.h>

namespace MCAsianOption
{
    inline int success(cudaError_t result)
    {
        return result == cudaSuccess;
    }

    MCAsianOptionPlan::MCAsianOptionPlan(int nOptions) :
        m_nOptions(nOptions),
        m_nScenarios(0),
        m_nTimesteps(0),
        m_spot(nOptions),
        m_strike(nOptions),
        m_r(nOptions),
        m_sigma(nOptions),
        m_tenor(nOptions),
        m_callPut(nOptions),
        m_devicePlan(0)
    {
    }

    MCAsianOptionPlan::MCAsianOptionPlan(int nOptions, float *spot, float *strike, float *r, float *sigma, float *tenor, CallPut *callPut) :
        m_nOptions(nOptions),
        m_nScenarios(0),
        m_nTimesteps(0),
        m_spot(spot, spot + nOptions),
        m_strike(strike, strike + nOptions),
        m_r(r, r + nOptions),
        m_sigma(sigma, sigma + nOptions),
        m_tenor(tenor, tenor + nOptions),
        m_callPut(callPut, callPut + nOptions),
        m_devicePlan(0)
    {
    }

    MCAsianOptionPlan::~MCAsianOptionPlan()
    {
        if (m_devicePlan)
        {
            destroyDevicePlan();
            m_devicePlan = 0;
        }
    }

    void MCAsianOptionPlan::setSpot(float *spot)
    {
        m_spot.assign(spot, spot + m_nOptions);
    }

    void MCAsianOptionPlan::setStrike(float *strike)
    {
        m_strike.assign(strike, strike + m_nOptions);
    }

    void MCAsianOptionPlan::setR(float *r)
    {
        m_r.assign(r, r + m_nOptions);
    }

    void MCAsianOptionPlan::setSigma(float *sigma)
    {
        m_sigma.assign(sigma, sigma + m_nOptions);
    }

    void MCAsianOptionPlan::setTenor(float *tenor)
    {
        m_tenor.assign(tenor, tenor + m_nOptions);
    }

    void MCAsianOptionPlan::setCallPut(CallPut *callPut)
    {
        m_callPut.assign(callPut, callPut + m_nOptions);
    }

    void MCAsianOptionPlan::setupMC(unsigned int nScenarios, unsigned int nTimesteps)
    {
        m_nScenarios = nScenarios;
        m_nTimesteps = nTimesteps;
    }

    unsigned int MCAsianOptionPlan::getNOptions(void)
    {
        return m_nOptions;
    }

    unsigned int MCAsianOptionPlan::getNScenarios(void)
    {
        return m_nScenarios;
    }

    unsigned int MCAsianOptionPlan::getNTimesteps(void)
    {
        return m_nTimesteps;
    }

    MCAsianOptionPlan::MCAsianOptionDevicePlan *MCAsianOptionPlan::createDevicePlan(void)
    {
        bool ok = true;

        if (m_nScenarios == 0 || m_nTimesteps == 0)
        {
            return NULL;
        }

        if (m_devicePlan)
        {
            return m_devicePlan;
        }

        // Allocate device memory
        float   *d_spot    = 0;
        float   *d_strike  = 0;
        float   *d_r       = 0;
        float   *d_sigma   = 0;
        float   *d_tenor   = 0;
        CallPut *d_callPut = 0;

        if (ok && ! success(cudaMalloc((void **)&d_spot, m_nOptions * sizeof(float))))
        {
            d_spot = 0;
            ok = false;
        }

        if (ok && ! success(cudaMalloc((void **)&d_strike, m_nOptions * sizeof(float))))
        {
            d_strike = 0;
            ok = false;
        }

        if (ok && ! success(cudaMalloc((void **)&d_r, m_nOptions * sizeof(float))))
        {
            d_r = 0;
            ok = false;
        }

        if (ok && ! success(cudaMalloc((void **)&d_sigma, m_nOptions * sizeof(float))))
        {
            d_sigma = 0;
            ok = false;
        }

        if (ok && ! success(cudaMalloc((void **)&d_tenor, m_nOptions * sizeof(float))))
        {
            d_tenor = 0;
            ok = false;
        }

        if (ok && ! success(cudaMalloc((void **)&d_callPut, m_nOptions * sizeof(CallPut))))
        {
            d_callPut = 0;
            ok = false;
        }

        // Copy data to device
        if (ok && ! success(cudaMemcpy(d_spot, &m_spot[0], m_nOptions * sizeof(float), cudaMemcpyHostToDevice)))
            ok = false;

        if (ok && ! success(cudaMemcpy(d_strike, &m_strike[0], m_nOptions * sizeof(float), cudaMemcpyHostToDevice)))
            ok = false;

        if (ok && ! success(cudaMemcpy(d_r, &m_r[0], m_nOptions * sizeof(float), cudaMemcpyHostToDevice)))
            ok = false;

        if (ok && ! success(cudaMemcpy(d_sigma, &m_sigma[0], m_nOptions * sizeof(float), cudaMemcpyHostToDevice)))
            ok = false;

        if (ok && ! success(cudaMemcpy(d_tenor, &m_tenor[0], m_nOptions * sizeof(float), cudaMemcpyHostToDevice)))
            ok = false;

        if (ok && ! success(cudaMemcpy(d_callPut, &m_callPut[0], m_nOptions * sizeof(CallPut), cudaMemcpyHostToDevice)))
            ok = false;

        // Create the plan on the host
        MCAsianOptionDevicePlan h_plan;
        h_plan.nOptions   = m_nOptions;
        h_plan.nScenarios = m_nScenarios;
        h_plan.nTimesteps = m_nTimesteps;
        h_plan.spot       = d_spot;
        h_plan.strike     = d_strike;
        h_plan.r          = d_r;
        h_plan.sigma      = d_sigma;
        h_plan.tenor      = d_tenor;
        h_plan.callPut    = d_callPut;

        // Copy the plan to the device
        if (ok && ! success(cudaMalloc((void **)&m_devicePlan, sizeof(MCAsianOptionDevicePlan))))
        {
            m_devicePlan = 0;
            ok = false;
        }

        if (ok && ! success(cudaMemcpy(m_devicePlan, &h_plan, sizeof(MCAsianOptionDevicePlan), cudaMemcpyHostToDevice)))
            ok = false;

        return m_devicePlan;
    }

    void MCAsianOptionPlan::destroyDevicePlan(void)
    {
        bool ok = true;

        if (m_devicePlan)
        {
            // Get a copy of the plan
            MCAsianOptionDevicePlan h_plan;
            h_plan.spot    = 0;
            h_plan.strike  = 0;
            h_plan.r       = 0;
            h_plan.sigma   = 0;
            h_plan.callPut = 0;

            if (ok && ! success(cudaMemcpy(&h_plan, m_devicePlan, sizeof(MCAsianOptionDevicePlan), cudaMemcpyDeviceToHost)))
                ok = false;

            // Cleanup
            if (h_plan.spot)
            {
                cudaFree(h_plan.spot);
                h_plan.spot = 0;
            }

            if (h_plan.strike)
            {
                cudaFree(h_plan.strike);
                h_plan.strike = 0;
            }

            if (h_plan.r)
            {
                cudaFree(h_plan.r);
                h_plan.r = 0;
            }

            if (h_plan.sigma)
            {
                cudaFree(h_plan.sigma);
                h_plan.sigma = 0;
            }

            if (h_plan.tenor)
            {
                cudaFree(h_plan.tenor);
                h_plan.tenor = 0;
            }

            if (h_plan.callPut)
            {
                cudaFree(h_plan.callPut);
                h_plan.callPut = 0;
            }

            if (m_devicePlan)
            {
                cudaFree(m_devicePlan);
                m_devicePlan = 0;
            }
        }
    }
}
