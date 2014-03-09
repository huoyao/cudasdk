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

namespace QRand
{
    class QRandN
    {
        public:
            QRandN(unsigned long nVectors, unsigned long nDimensions = 1);
            virtual ~QRandN();

            float *getDeviceMutable(void);

        protected:
            unsigned long  m_nDimensions;
            unsigned long  m_nVectors;
            float         *m_matrix;

        private:
            void generateMatrix(void);
    };
}
