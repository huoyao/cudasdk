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

#ifndef NV_VIDEO_ENCODER
#define NV_VIDEO_ENCODER

// This preprocessor definition is required for the use case where we want to use
// GPU device memory input with the CUDA H.264 Encoder.  This define is needed because
// 64-bit CUdevicePtrs are not supported in R260 drivers with NVCUVENC.  With driver versions
// after R265, ?UdevicePtrs are supported with NVCUVENC.  CUDA kernels that want to interop
// with the CUDA H.264 Encoder, this define must also be present for drivers <= R260.
//#define CUDA_FORCE_API_VERSION 3010

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "unknwn.h"
#include <math.h> // log10
#include <nvcuvid.h> // this is for the Cuvideoctxlock
#include "NVEncoderAPI.h"
#include "NVEncodeDataTypes.h"
#include "types.h"


// Wrapper class around the CUDA video encoder API.
//
class VideoEncoder
{
    public:
        VideoEncoder(NVEncoderParams *pParams, bool bUseDeviceMem = false);
        ~VideoEncoder();

        // General high level initialization and parameter settings
        bool InitEncoder(NVEncoderParams *pParams);
        bool SetEncodeParameters(NVEncoderParams *pParams);
        bool SetCBFunctions(NVVE_CallbackParams *pCallback, void *pUserData = NULL);
        bool CreateHWEncoder(NVEncoderParams *pParams);

        // Help functions for NVCUVENC
        bool GetEncodeParamsConfig(NVEncoderParams *pParams);
        bool SetCodecType(NVEncoderParams *pParams);
        bool SetParameters(NVEncoderParams *pParams);
        int  DisplayGPUCaps(int deviceOrdinal, NVEncoderParams *pParams, bool bDisplay);
        int  GetGPUCount(NVEncoderParams *pParams, int *gpuPerf, int *bestGPU);
        void SetActiveGPU(NVEncoderParams *pParams, int gpuID);
        void SetGPUOffloadLevel(NVEncoderParams *pParams);

        // These are for setting and getting parameter values
        HRESULT GetParamValue(DWORD dwParamType, void *pData);
        HRESULT SetParamValue(DWORD dwParamType, void *pData);

        // Functions to start, stop, and encode frames
        void  Start();
        void  Stop();
        size_t ReadNextFrame();

        void  CopyUYVYorYUY2Frame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock);
        void  CopyYV12orIYUVFrame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock);
        void  CopyNV12Frame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock);
        bool  EncodeFrame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock);

    public:
        unsigned char *GetVideoFrame()
        {
            return m_pVideoFrame;
        }
        unsigned char *GetCharBuf()
        {
            return m_pCharBuf;
        }

        FILE *fileIn()
        {
            return fpIn;
        }
        FILE *fileOut()
        {
            return fpOut;
        }
        FILE *fileConfig()
        {
            return fpConfig;
        }
        void resetFrameCount()
        {
            m_nFrameCount = 0;
            m_nLastFrameNumber = 0;
            m_bLastFrame  = false;
            m_bEncodeDone = false;
            m_lFrameSummation = 0;
        }
        long GetLastFrameNumber()
        {
            return m_nLastFrameNumber;
        }
        long GetBytesPerFrame()
        {
            return m_nVideoFrameSize;
        }
        void SetEncodeDone()
        {
            m_bEncodeDone = true;
        }
        bool IsLastFrameSent()
        {
            return m_bLastFrame;
        }
        bool IsEncodeDone()
        {
            return m_bEncodeDone;
        }
        DWORD frameCount()
        {
            return m_nFrameCount;
        }
        void incFrameCount()
        {
            m_nFrameCount++;
        }
        void setMSE(double *mse)
        {
            m_MSE[0] = mse[0];
            m_MSE[1] = mse[1];
            m_MSE[2] = mse[2];
        }
        // because frames have a unique #, we keep adding the frame to a running count
        // to check to ensure that every last frame is completed, we will subtract or add
        // to the running count.  If frameSummation == 0, then we have reached the last value.
        void frameSummation(long frame_num)
        {
            m_lFrameSummation += frame_num;
        }

        long getFrameSum()
        {
            return m_lFrameSummation;
        }

    private:
        void             *m_pEncoder;
        void             *m_pSNRData;
        unsigned char    *m_pVideoFrame;
        unsigned char    *m_pCharBuf;
        unsigned int    m_nVideoFrameSize;

        bool            m_bLastFrame, m_bEncodeDone;
        long            m_lFrameSummation; // this is summation of the frame all frames
        long            m_nFrameCount, m_nLastFrameNumber;

        NVEncoderParams *m_pEncoderParams;
        NVVE_CallbackParams m_NVCB;

        FILE *fpIn, *fpOut, *fpConfig;

        LARGE_INTEGER m_liUserTime0, m_liKernelTime0;
        DWORD m_dwStartTime;

        double m_MSE[3];
};

#endif // NV_VIDEO_ENCODER