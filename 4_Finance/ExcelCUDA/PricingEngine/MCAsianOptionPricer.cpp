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

#include "MCAsianOptionPricer.h"
#include "MCAsianOptionPricerCore.h"
#include "MCAsianOptionPlan.h"

namespace MCAsianOption
{
    MCAsianOptionPricer::MCAsianOptionPricer(MCAsianOptionPlan *plan) :
        m_plan(plan)
    {
    }

    MCAsianOptionPricer::~MCAsianOptionPricer()
    {
    }

    void MCAsianOptionPricer::execute(float *result)
    {
        MCAsianOptionPlan::MCAsianOptionDevicePlan *d_plan = m_plan->createDevicePlan();
        MCAsianOptionPricerCore(m_plan, d_plan, result);
    }

    void MCAsianOptionPricer::operator()(float *result)
    {
        execute(result);
    }
}