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

#include "PricingEngine.h"

#include "MCAsianOptionPlan.h"
#include "MCAsianOptionPricer.h"

void priceAsianOptions(float *spot,
                       float *strike,
                       float *r,
                       float *sigma,
                       float *tenor,
                       int *callNotPut,
                       float *result,
                       unsigned int nOptions,
                       unsigned int nTimesteps,
                       unsigned int nScenarios)
{
    MCAsianOption::MCAsianOptionPlan   plan(nOptions, spot, strike, r, sigma, tenor, (MCAsianOption::MCAsianOptionPlan::CallPut *)callNotPut);
    plan.setupMC(nScenarios, nTimesteps);
    MCAsianOption::MCAsianOptionPricer pricer(&plan);

    pricer(result);
}
