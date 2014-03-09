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

#include <vector>

namespace MCAsianOption
{
    class MCAsianOptionPlan
    {
        public:
            enum CallPut {Put = 0, Call = 1};

            MCAsianOptionPlan(int nOptions);
            MCAsianOptionPlan(int nOptions, float *spot, float *strike, float *r, float *sigma, float *tenor, CallPut *callPut);
            virtual ~MCAsianOptionPlan();

            void setSpot(float *spot);
            void setStrike(float *strike);
            void setR(float *r);
            void setSigma(float *sigma);
            void setTenor(float *tenor);
            void setCallPut(CallPut *callPut);

            void setupMC(unsigned int nScenarios, unsigned int nTimesteps);

            unsigned int getNOptions(void);
            unsigned int getNScenarios(void);
            unsigned int getNTimesteps(void);

            struct MCAsianOptionDevicePlan
            {
                float   *spot;
                float   *strike;
                float   *r;
                float   *sigma;
                float   *tenor;
                CallPut *callPut;
                unsigned int nOptions;
                unsigned int nScenarios;
                unsigned int nTimesteps;
            };

            MCAsianOptionDevicePlan *createDevicePlan(void);

            //protected:
            unsigned int m_nOptions;
            unsigned int m_nScenarios;
            unsigned int m_nTimesteps;

            std::vector<float>   m_spot;
            std::vector<float>   m_strike;
            std::vector<float>   m_r;
            std::vector<float>   m_sigma;
            std::vector<float>   m_tenor;
            std::vector<CallPut> m_callPut;

            void destroyDevicePlan();

        private:
            MCAsianOptionDevicePlan *m_devicePlan;
    };
}
