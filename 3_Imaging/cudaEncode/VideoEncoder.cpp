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

// This preprocessor definition is required for the use case where we want to use
// GPU device memory input with the CUDA H.264 Encoder.  This define is needed because
// 64-bit CUdevicePtrs are not supported in R260 drivers with NVCUVENC.  With driver versions
// after R265, ?UdevicePtrs are supported with NVCUVENC.  CUDA kernels that want to interop
// with the CUDA H.264 Encoder, this define must also be present for drivers <= R260.
//#define CUDA_FORCE_API_VERSION 3010

#include "VideoEncoder.h"

#include <cstring>
#include <cassert>

// includes, CUDA
#include <cuda.h>
#include <builtin_types.h>
#include <drvapi_error_string.h>

#ifdef _DEBUG
#define PRINTF(x) printf((x))
#else
#define PRINTF(x)
#endif

#ifndef MAX
#define MAX(a,b) (a > b) ? a : b
#endif

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
    if (CUDA_SUCCESS != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
    return nGpuArchCoresPerSM[7].Cores;
}
// end of GPU Architecture definitions
// end of CUDA Helper Functions

// We have a global pointer to the Video Encoder
VideoEncoder   *g_pVideoEncoder = NULL;

// Error message handling
inline void CheckNVEncodeError(HRESULT hr_error, const char *NVfunction, const char *error_string)
{
    if (FAILED(hr_error))
    {
        printf("%s(%s) error: 0x%08x\n", NVfunction, error_string, hr_error);
    }
}


// NVCUVENC callback function to signal the start of bitstream that is to be encoded
static unsigned char *_stdcall NVDefault_HandleAcquireBitStream(int *pBufferSize, void *pUserdata)
{
    *pBufferSize = 1024*1024;
    return g_pVideoEncoder->GetCharBuf();
}

//NVCUVENC callback function to signal that the encoded bitstream is ready to be written to file
static void _stdcall NVDefault_HandleReleaseBitStream(int nBytesInBuffer, unsigned char *cb,void *pUserdata)
{
    if (g_pVideoEncoder && g_pVideoEncoder->fileOut())
    {
        fwrite(cb,1,nBytesInBuffer,g_pVideoEncoder->fileOut());
    }

    return;
}

//NVCUVENC callback function to signal that the encoding operation on the frame has begun
static void _stdcall NVDefault_HandleOnBeginFrame(const NVVE_BeginFrameInfo *pbfi, void *pUserdata)
{
    return;
}

//NVCUVENC callback function signals that the encoding operation on the frame has ended
static void _stdcall NVDefault_HandleOnEndFrame(const NVVE_EndFrameInfo *pefi, void *pUserdata)
{
    double psnr[3], mse[3] = { 0.0, 0.0, 0.0 };

    if (pUserdata)
    {
        memcpy(psnr, pUserdata,sizeof(psnr));
    }

    mse[0] += psnr[0];
    mse[1] += psnr[1];
    mse[2] += psnr[2];
    g_pVideoEncoder->setMSE(mse);

    g_pVideoEncoder->frameSummation(-(pefi->nFrameNumber));

    if (g_pVideoEncoder->IsLastFrameSent())
    {
        // Check to see if the last frame has been sent
        if (g_pVideoEncoder->getFrameSum() == 0)
        {
            printf(">> Encoder has finished encoding last frame\n<< n");
        }
    }
    else
    {
#ifdef _DEBUG
        printf("HandleOnEndFrame (%d), FrameCount (%d), FrameSummation (%d)\n",
               pefi->nFrameNumber, g_pVideoEncoder->frameCount(),
               g_pVideoEncoder->getFrameSum());
#endif
    }

    return;
}


// Helper functions to measure overall performance
BOOL StartPerfLog(DWORD *pdwElapsedTime, LARGE_INTEGER *pliUserTime0, LARGE_INTEGER *pliKernelTime0)
{
    *pdwElapsedTime = timeGetTime();

    FILETIME ftCreationTime, ftExitTime, ftKernelTime, ftUserTime;

    g_pVideoEncoder->resetFrameCount();

    if (GetProcessTimes(GetCurrentProcess(), &ftCreationTime, &ftExitTime, &ftKernelTime, &ftUserTime))
    {
        pliUserTime0->LowPart = ftUserTime.dwLowDateTime;
        pliUserTime0->HighPart = ftUserTime.dwHighDateTime;
        pliKernelTime0->LowPart = ftKernelTime.dwLowDateTime;
        pliKernelTime0->HighPart = ftKernelTime.dwHighDateTime;
    }
    else
    {
        printf("\nerror in measuring CPU utilization\n");
    }

    return TRUE;
}

// Helper functions to measure overall performance
BOOL StopPerfLog(DWORD dwStartTime, LARGE_INTEGER liUserTime0, LARGE_INTEGER liKernelTime0, void *myEncoder)
{
    HRESULT hr = S_OK;

    LONGLONG lNumCodedFrames    = 0;
    DWORD dwElapsedTime         = 0;

    dwElapsedTime = timeGetTime() - dwStartTime;

    DWORD hh, mm, ss, ms;
    ms = dwElapsedTime;
    ss = (ms / 1000);
    mm = (ss / 60);
    hh = (mm / 60);

    printf("\n");
    printf("[H.264 Encoding Statistics]\n");
    printf("\tNumber of Coded Frames     : %d\n", g_pVideoEncoder->frameCount());
    printf("\tElapsed time (hh:mm:ss:ms) : %02d:%02d:%02d.%03d \n", hh%24, mm%60, ss%60, ms%1000);
    printf("\tAverage FPS (end to end)   : %f\n", (float)((double)g_pVideoEncoder->frameCount()/((double)dwElapsedTime/1000.0f)));

    //CPU utilization
    FILETIME ftCreationTime, ftExitTime, ftKernelTime, ftUserTime;

    if (GetProcessTimes(GetCurrentProcess(), &ftCreationTime, &ftExitTime, &ftKernelTime, &ftUserTime))
    {
        LARGE_INTEGER liUserTime, liKernelTime;
        SYSTEM_INFO si;
        LONG nCores;

        memset(&si, 0, sizeof(si));
        GetSystemInfo(&si);
        nCores = MAX(si.dwNumberOfProcessors, 1);
        liUserTime.LowPart = ftUserTime.dwLowDateTime;
        liUserTime.HighPart = ftUserTime.dwHighDateTime;
        liKernelTime.LowPart = ftKernelTime.dwLowDateTime;
        liKernelTime.HighPart = ftKernelTime.dwHighDateTime;
        printf("\tCPU utilization (%d cores)  : %5.2f%% (user:%5.2f%%, kernel:%5.2f%%)\n", nCores,
               ((double)((liUserTime.QuadPart - liUserTime0.QuadPart)+(liKernelTime.QuadPart - liKernelTime0.QuadPart)) / (100*nCores)) / (double)dwElapsedTime,
               ((double)(liUserTime.QuadPart - liUserTime0.QuadPart) / 100) / (double)dwElapsedTime,
               ((double)(liKernelTime.QuadPart - liKernelTime0.QuadPart) / 100) / (double)dwElapsedTime);
    }
    else
    {
        printf("\nerror in measuring CPU utilization\n");
    }

    return TRUE;
}

VideoEncoder::VideoEncoder(NVEncoderParams *pParams, bool bUseDeviceMem)
    : m_pEncoderParams(pParams),
      m_pSNRData(NULL),
      m_pVideoFrame(NULL),
      m_pCharBuf(NULL),
      m_nLastFrameNumber(0),
      m_nFrameCount(0),
      m_lFrameSummation(0),
      m_bLastFrame(false),
      m_bEncodeDone(false)
{
    HRESULT hr = S_OK;

    g_pVideoEncoder = this;
    m_MSE[0] = 0.0;
    m_MSE[1] = 0.0;
    m_MSE[2] = 0.0;

    printf("Configuration  file: <%s>\n", m_pEncoderParams->configFile);
    printf("Source  input  file: <%s>\n", m_pEncoderParams->inputFile);
    printf("Encoded output file: <%s>\n", m_pEncoderParams->outputFile);
    printf("Measurement: %s\n\n", "(FPS) Frames Per Second");

    if ((fpIn = fopen(m_pEncoderParams->inputFile,"rb")) == NULL)
    {
        printf("VideoEncoder() - fopen() error! - The input file \"%s\" could not be opened for reading\n", m_pEncoderParams->inputFile);
        assert(0);
    }

    if ((fpOut = fopen(m_pEncoderParams->outputFile,"wb")) == NULL)
    {
        printf("VideoEncoder() - fopen() error - The output file \"%s\" could not be created for writing\n", m_pEncoderParams->outputFile);
        assert(0);
    }

    // Get the encoding parameters
    if (GetEncodeParamsConfig(m_pEncoderParams) != true)
    {
        printf("\nGetEncodeParamsConfig() error!\n");
        assert(0);
    }

    // If we want to force device memory as input, we can do so
    if (bUseDeviceMem)
    {
        m_pEncoderParams->iUseDeviceMem = 1;
        pParams->iUseDeviceMem = 1;
    }
}


VideoEncoder::~VideoEncoder()
{
    if (fpIn)
    {
        fclose(fpIn);
    }

    if (fpOut)
    {
        fclose(fpOut);
    }

    if (fpConfig)
    {
        fclose(fpConfig);
    }

    if (m_pEncoder)
    {
        NVDestroyEncoder(m_pEncoder);
        m_pEncoder = NULL;
    }

    // clear the global pointer
    g_pVideoEncoder = NULL;

    if (m_pVideoFrame)
    {
        delete [] m_pVideoFrame;
    }

    if (m_pCharBuf)
    {
        delete [] m_pCharBuf;
    }
}

bool
VideoEncoder::InitEncoder(NVEncoderParams *pParams)
{
    // Create the Encoder API Interface
    HRESULT hr = NVCreateEncoder(&m_pEncoder);
    printf("VideoEncoder() - NVCreateEncoder <%s>, hr = %08x\n", (FAILED(hr) ? "FAILED!" : "SUCCESS!"), hr);

    if (FAILED(hr))
    {
        return false;
    }

    // Note before we can set the GPU or query the GPU we wish to encode, it is necessary to set the CODECS type
    // This must be set before we can call any GetParamValue, otherwise it will not succeed
    SetCodecType(m_pEncoderParams);

    // Query the GPUs available for encoding
    int gpuPerf = 0, bestGPU = 0;
    GetGPUCount(m_pEncoderParams, &gpuPerf, &bestGPU);

    if (m_pEncoderParams->force_device)
    {
        SetActiveGPU(m_pEncoderParams, m_pEncoderParams->iForcedGPU);
    }
    else
    {
        m_pEncoderParams->iForcedGPU = bestGPU;
        SetActiveGPU(m_pEncoderParams, m_pEncoderParams->iForcedGPU);
    }

    printf("    YUV Format: %s [%s] (%d-bpp)\n", sSurfaceFormat[pParams->iSurfaceFormat].name,
           sSurfaceFormat[pParams->iSurfaceFormat].yuv_type,
           sSurfaceFormat[pParams->iSurfaceFormat].bpp);
    printf("    Frame Type: %s\n\n",           sPictureStructure[pParams->iPictureType]);

    printf(" >> Video Input: %s memory\n",     m_pEncoderParams->iUseDeviceMem ? "GPU Device" : "CPU System");

    m_pCharBuf = new unsigned char[1024*1024];

    pParams->GPU_count = m_pEncoderParams->GPU_count;
    return true;
}

bool
VideoEncoder::SetEncodeParameters(NVEncoderParams *pParams)
{
    //  Allocate a little bit more memory, in case we have higher formats to handle
    m_nVideoFrameSize = ((pParams->iInputSize[0] * pParams->iInputSize[1] *
                          sSurfaceFormat[pParams->iSurfaceFormat].bpp) / 8) * sizeof(char);
    m_pVideoFrame = new unsigned char[m_nVideoFrameSize];

    // Set the GPU Offload Level
    SetGPUOffloadLevel(m_pEncoderParams);

    // Now Set the Encoding Parameters
    bool bRetVal = SetParameters(m_pEncoderParams);
    printf("VideoEncoder() - SetEncodeParameters <%s>, bRetVal = %08x\n", ((bRetVal != TRUE) ? "FAILED!" : "SUCCESS!"), bRetVal);

    return bRetVal;
}

bool
VideoEncoder::SetCBFunctions(NVVE_CallbackParams *pCB, void *pUserData)
{
    if (pCB)
    {
        // Copy NVIDIA callback functions
        m_NVCB = *pCB;
        //Register the callback structure functions
        NVRegisterCB(m_pEncoder, m_NVCB, pUserData); //register the callback structure
    }
    else
    {
        // We use the callback functions defined in this class
        memset(&m_NVCB,0,sizeof(NVVE_CallbackParams));
        m_NVCB.pfnacquirebitstream = NVDefault_HandleAcquireBitStream;
        m_NVCB.pfnonbeginframe     = NVDefault_HandleOnBeginFrame;
        m_NVCB.pfnonendframe       = NVDefault_HandleOnEndFrame;
        m_NVCB.pfnreleasebitstream = NVDefault_HandleReleaseBitStream;

        //Register the callback structure functions
        (m_pEncoder, m_NVCB, m_pSNRData); //register the callback structure
    }

    printf("VideoEncoder() - SetCBFunctions <SUCCESS>\n");
    return true;
}

bool
VideoEncoder::CreateHWEncoder(NVEncoderParams *pParams)
{
    // Create the NVIDIA HW resources for Encoding on NVIDIA hardware
    HRESULT hr = NVCreateHWEncoder(m_pEncoder);
    printf("VideoEncoder() - NVCreateHWEncoder <%s>, hr = %08x\n", (FAILED(hr) ? "FAILED!  Unable to create NVIDIA HW Video Encoder" : "OK!"), hr);

    if (FAILED(hr))
    {
        return false;
    }

    unsigned char buf2[10];
    int size;
    hr = NVGetSPSPPS(m_pEncoder, buf2, 10, &size);

    if (FAILED(hr))
    {
        printf("\nNVGetSPSPPS() error getting SPSPPS buffer \n");
    }
    else
    {
        printf("VideoEncoder() - NVGetSPSPPS <%s>, hr = %08x\n", (FAILED(hr) ? "FAILED!" : "OK!"), hr);
    }

    return (FAILED(hr) ? false : true);
}


void
VideoEncoder::Start()
{
    printf("\n[VideoEncoder <Starting>]\n");
    printf("  [input]  = %s\n", m_pEncoderParams->inputFile);
    printf("  [output] = %s\n", m_pEncoderParams->outputFile);

    StartPerfLog(&m_dwStartTime, &m_liUserTime0, &m_liKernelTime0); //calculate fps and cpu utilization
}

void
VideoEncoder::Stop()
{
    printf("\n[VideoEncoder() <Stopped>]\n");
    printf("  [input]  = %s\n", m_pEncoderParams->inputFile);
    printf("  [output] = %s\n", m_pEncoderParams->outputFile);

    StopPerfLog(m_dwStartTime, m_liUserTime0, m_liKernelTime0, m_pEncoder);
}

// Function for handling reading in subsequent frames
size_t
VideoEncoder::ReadNextFrame()
{
    size_t bytes_read = fread(m_pVideoFrame, 1, m_nVideoFrameSize, fileIn());

    if (!feof(fileIn()))
    {
        if (bytes_read != m_nVideoFrameSize)
        {
            printf("ReadNextFrame() bytes_read = %d mismatches with VideoFrameSize = %d\n", bytes_read, m_nVideoFrameSize);
            return -1;
        }
    }

    return bytes_read;
}

// UYVY/YUY2 are both 4:2:2 formats (16bpc)
// Luma, U, V are interleaved, chroma is subsampled (w/2,h)
void
VideoEncoder::CopyUYVYorYUY2Frame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock)
{
    // Source is YUVY/YUY2 4:2:2, the YUV data in a packed and interleaved
    // YUV Copy setup
    CUDA_MEMCPY2D stCopyYUV422;
    memset((void *)&stCopyYUV422, 0, sizeof(stCopyYUV422));
    stCopyYUV422.srcXInBytes          = 0;
    stCopyYUV422.srcY                 = 0;
    stCopyYUV422.srcMemoryType        = CU_MEMORYTYPE_HOST;
    stCopyYUV422.srcHost              = sFrameParams.picBuf;
    stCopyYUV422.srcDevice            = 0;
    stCopyYUV422.srcArray             = 0;
    stCopyYUV422.srcPitch             = sFrameParams.Width * 2;

    stCopyYUV422.dstXInBytes          = 0;
    stCopyYUV422.dstY                 = 0;
    stCopyYUV422.dstMemoryType        = CU_MEMORYTYPE_DEVICE;
    stCopyYUV422.dstHost              = 0;
    stCopyYUV422.dstDevice            = dptr_VideoFrame;
    stCopyYUV422.dstArray             = 0;
    stCopyYUV422.dstPitch             = m_pEncoderParams->nDeviceMemPitch;

    stCopyYUV422.WidthInBytes         = m_pEncoderParams->iInputSize[0]*2;
    stCopyYUV422.Height               = m_pEncoderParams->iInputSize[1];

    // Don't forget we need to lock/unlock between memcopies
    checkCudaErrors(cuvidCtxLock(ctxLock, 0));
    checkCudaErrors(cuMemcpy2D(&stCopyYUV422));     // Now DMA Luma/Chroma
    checkCudaErrors(cuvidCtxUnlock(ctxLock, 0));
}

// YV12/IYUV are both 4:2:0 planar formats (12bpc)
// Luma, U, V chroma planar (12bpc), chroma is subsampled (w/2,h/2)
void
VideoEncoder::CopyYV12orIYUVFrame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock)
{
    // Source is YV12/IYUV, this native format is converted to NV12 format by the video encoder
    // (1) luma copy setup
    CUDA_MEMCPY2D stCopyLuma;
    memset((void *)&stCopyLuma, 0, sizeof(stCopyLuma));
    stCopyLuma.srcXInBytes          = 0;
    stCopyLuma.srcY                 = 0;
    stCopyLuma.srcMemoryType        = CU_MEMORYTYPE_HOST;
    stCopyLuma.srcHost              = sFrameParams.picBuf;
    stCopyLuma.srcDevice            = 0;
    stCopyLuma.srcArray             = 0;
    stCopyLuma.srcPitch             = sFrameParams.Width;

    stCopyLuma.dstXInBytes          = 0;
    stCopyLuma.dstY                 = 0;
    stCopyLuma.dstMemoryType        = CU_MEMORYTYPE_DEVICE;
    stCopyLuma.dstHost              = 0;
    stCopyLuma.dstDevice            = dptr_VideoFrame;
    stCopyLuma.dstArray             = 0;
    stCopyLuma.dstPitch             = m_pEncoderParams->nDeviceMemPitch;

    stCopyLuma.WidthInBytes         = m_pEncoderParams->iInputSize[0];
    stCopyLuma.Height               = m_pEncoderParams->iInputSize[1];

    // (2) chroma copy setup, U/V can be done together
    CUDA_MEMCPY2D stCopyChroma;
    memset((void *)&stCopyChroma, 0, sizeof(stCopyChroma));
    stCopyChroma.srcXInBytes        = 0;
    stCopyChroma.srcY               = m_pEncoderParams->iInputSize[1]<<1; // U/V chroma offset
    stCopyChroma.srcMemoryType      = CU_MEMORYTYPE_HOST;
    stCopyChroma.srcHost            = sFrameParams.picBuf;
    stCopyChroma.srcDevice          = 0;
    stCopyChroma.srcArray           = 0;
    stCopyChroma.srcPitch           = sFrameParams.Width>>1; // chroma is subsampled by 2 (but it has U/V are next to each other)

    stCopyChroma.dstXInBytes        = 0;
    stCopyChroma.dstY               = m_pEncoderParams->iInputSize[1]<<1; // chroma offset (srcY*srcPitch now points to the chroma planes)
    stCopyChroma.dstMemoryType      = CU_MEMORYTYPE_DEVICE;
    stCopyChroma.dstHost            = 0;
    stCopyChroma.dstDevice          = dptr_VideoFrame;
    stCopyChroma.dstArray           = 0;
    stCopyChroma.dstPitch           = m_pEncoderParams->nDeviceMemPitch>>1;

    stCopyChroma.WidthInBytes       = m_pEncoderParams->iInputSize[0]>>1;
    stCopyChroma.Height             = m_pEncoderParams->iInputSize[1]; // U/V are sent together

    // Don't forget we need to lock/unlock between memcopies
    checkCudaErrors(cuvidCtxLock(ctxLock, 0));
    checkCudaErrors(cuMemcpy2D(&stCopyLuma));       // Now DMA Luma
    checkCudaErrors(cuMemcpy2D(&stCopyChroma));     // Now DMA Chroma channels (UV side by side)
    checkCudaErrors(cuvidCtxUnlock(ctxLock, 0));
}

// NV12 is 4:2:0 format (12bpc)
// Luma followed by U/V chroma interleaved (12bpc), chroma is subsampled (w/2,h/2)
void
VideoEncoder::CopyNV12Frame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock)
{
    // Source is NV12 in pitch linear memory
    // Because we are assume input is NV12 (if we take input in the native format), the encoder handles NV12 as a native format in pitch linear memory
    // Luma/Chroma can be done in a single transfer
    CUDA_MEMCPY2D stCopyNV12;
    memset((void *)&stCopyNV12, 0, sizeof(stCopyNV12));
    stCopyNV12.srcXInBytes          = 0;
    stCopyNV12.srcY                 = 0;
    stCopyNV12.srcMemoryType        = CU_MEMORYTYPE_HOST;
    stCopyNV12.srcHost              = sFrameParams.picBuf;
    stCopyNV12.srcDevice            = 0;
    stCopyNV12.srcArray             = 0;
    stCopyNV12.srcPitch             = sFrameParams.Width;

    stCopyNV12.dstXInBytes          = 0;
    stCopyNV12.dstY                 = 0;
    stCopyNV12.dstMemoryType        = CU_MEMORYTYPE_DEVICE;
    stCopyNV12.dstHost              = 0;
    stCopyNV12.dstDevice            = dptr_VideoFrame;
    stCopyNV12.dstArray             = 0;
    stCopyNV12.dstPitch             = m_pEncoderParams->nDeviceMemPitch;

    stCopyNV12.WidthInBytes         = m_pEncoderParams->iInputSize[0];
    stCopyNV12.Height               =(m_pEncoderParams->iInputSize[1] * 3) >> 1;

    // Don't forget we need to lock/unlock between memcopies
    checkCudaErrors(cuvidCtxLock(ctxLock, 0));
    checkCudaErrors(cuMemcpy2D(&stCopyNV12));    // Now DMA Luma/Chroma
    checkCudaErrors(cuvidCtxUnlock(ctxLock, 0));
}

// If dptr_VideoFrame is != 0, then this is from Device Memory.
// Otherwise we will assume that video is coming from system host memory
bool
VideoEncoder::EncodeFrame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock)
{
    // If this is the first frame, we can start timing
    if (m_nFrameCount == 0)
    {
        Start();
    }

    HRESULT hr = S_OK;

    if (m_pEncoderParams->iUseDeviceMem)
    {
        // Copies video frame from system memory, and passes it as a System pointer to the API
        switch (m_pEncoderParams->iSurfaceFormat)
        {
            case UYVY: // UYVY (4:2:2)
            case YUY2:  // YUY2 (4:2:2)
                CopyUYVYorYUY2Frame(sFrameParams, dptr_VideoFrame, ctxLock);
                break;

            case YV12: // YV12 (4:2:0), Y V U
            case IYUV: // IYUV (4:2:0), Y U V
                CopyYV12orIYUVFrame(sFrameParams, dptr_VideoFrame, ctxLock);
                break;

            case NV12: // NV12 (4:2:0)
                CopyNV12Frame(sFrameParams, dptr_VideoFrame, ctxLock);
                break;

            default:
                break;
        }

        sFrameParams.picBuf = NULL;  // Must be set to NULL in order to support device memory input
        hr = NVEncodeFrame(m_pEncoder, &sFrameParams, 0, (void *)dptr_VideoFrame); //send the video (device memory) to the
    }
    else
    {
        // Copies video frame from system memory, and passes it as a System pointer to the API
        hr = NVEncodeFrame(m_pEncoder, &sFrameParams, 0, m_pSNRData);
    }

    if (FAILED(hr))
    {
        printf("VideoEncoder::EncodeFrame() error when encoding frame (%d)\n", m_nFrameCount);
        return false;
    }

    if (sFrameParams.bLast)
    {
        m_bLastFrame  = true;
        m_nLastFrameNumber = m_nFrameCount;
    }
    else
    {
        frameSummation(m_nFrameCount);
        m_nFrameCount++;
    }

    return true;
}

HRESULT
VideoEncoder::GetParamValue(DWORD dwParamType, void *pData)
{
    HRESULT hr = S_OK;
    hr = NVGetParamValue(m_pEncoder, dwParamType, pData);

    if (hr != S_OK)
    {
        printf("  NVGetParamValue FAIL!: hr = %08x\n", hr);
    }

    return hr;
}

bool
VideoEncoder::GetEncodeParamsConfig(NVEncoderParams *pParams)
{
    assert(pParams != NULL);

    int iAspectRatio = 0;

    if (pParams == NULL)
    {
        return false;
    }

    fopen_s(&fpConfig, pParams->configFile, "r");

    if (fpConfig == NULL)
    {
        return false;
    }

    //read the params
    _flushall();
    char cTempArr[250];
    fscanf_s(fpConfig, "%d", &(pParams->iCodecType));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iOutputSize[0]));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iOutputSize[1]));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iInputSize[0]));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iInputSize[1]));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(iAspectRatio));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->Fieldmode));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iP_Interval));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iIDR_Period));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iDynamicGOP));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->RCType));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iAvgBitrate));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iPeakBitrate));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iQP_Level_Intra));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iQP_Level_InterP));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iQP_Level_InterB));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iFrameRate[0]));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iFrameRate[1]));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iDeblockMode));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iProfileLevel));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iForceIntra));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iForceIDR));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iClearStat));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->DIMode));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->Presets));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iDisableCabac));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iNaluFramingType));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iDisableSPSPPS));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);
    fscanf_s(fpConfig, "%d", &(pParams->iSliceCnt));
    fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

    switch (iAspectRatio)
    {
        case 0:
            {
                pParams->iAspectRatio[0] = 4;
                pParams->iAspectRatio[1] = 3;
            }
            break;

        case 1:
            {
                pParams->iAspectRatio[0] = 16;
                pParams->iAspectRatio[1] = 9;
            }
            break;

        case 2:
            {
                pParams->iAspectRatio[0] = 1;
                pParams->iAspectRatio[1] = 1;
            }
            break;

        default:
            {
                pParams->iAspectRatio[0] = 4;
                pParams->iAspectRatio[1] = 3;
            }
    }

    pParams->iAspectRatio[2] = 0;

    if (fpConfig)
    {
        fclose(fpConfig);
    }

    return true;
}

HRESULT
VideoEncoder::SetParamValue(DWORD dwParamType, void *pData)
{
    HRESULT hr = S_OK;
    hr = NVSetParamValue(m_pEncoder, dwParamType, pData);

    if (hr != S_OK)
    {
        printf("  NVSetParamValue: %26s,  hr = %08x ", sNVVE_EncodeParams[dwParamType].name, hr);

        for (int i=0; i < sNVVE_EncodeParams[dwParamType].params; i++)
        {
            printf(" %8d", *((DWORD *)pData+i));
            printf(", ");
        }

        printf(" FAILED!");
    }
    else
    {
        printf("  NVSetParamValue : %26s = ", sNVVE_EncodeParams[dwParamType].name);

        for (int i=0; i < sNVVE_EncodeParams[dwParamType].params; i++)
        {
            printf("%8d", *((DWORD *)pData+i));
            printf(", ");
        }
    }

    switch (dwParamType)
    {
        case NVVE_PROFILE_LEVEL:
            printf(" [%s/%s] ",
                   sProfileIDX2Char(sProfileName, (*((DWORD *)pData) & 0x00ff)),
                   sProfileIDX2Char(sProfileLevel, (*((DWORD *)pData) >> 8) & 0x00ff));
            break;

        case NVVE_FIELD_ENC_MODE:
            printf(" [%s]", sPictureType[*((DWORD *)pData)]);
            break;

        case NVVE_RC_TYPE:
            printf(" [%s]", sNVVE_RateCtrlType[*((DWORD *)pData)]);
            break;

        case NVVE_PRESETS:
            printf(" [%s Profile]", sVideoEncodePresets[*((DWORD *)pData)]);
            break;

        case NVVE_GPU_OFFLOAD_LEVEL:
            switch (*((DWORD *)pData))
            {
                case NVVE_GPU_OFFLOAD_DEFAULT:
                    printf(" [%s]", sGPUOffloadLevel[0]);
                    break;

                case NVVE_GPU_OFFLOAD_ESTIMATORS :
                    printf(" [%s]", sGPUOffloadLevel[1]);
                    break;

                case 16:
                    printf(" [%s]", sGPUOffloadLevel[2]);
                    break;
            }

            break;
    }

    printf("\n");
    return hr;
}

bool
VideoEncoder::SetCodecType(NVEncoderParams *pParams)
{
    assert(pParams != NULL);

    HRESULT hr = S_OK;
    hr = NVSetCodec(m_pEncoder, pParams->iCodecType);

    if (hr!=S_OK)
    {
        printf("\nSetCodecType FAIL\n");
        return false;
    }

    printf("  NVSetCodec ");

    if (pParams->iCodecType == 4)
    {
        printf("<H.264 Video>\n");
    }
    else if (pParams->iCodecType == 5)
    {
        // Support for this codec is being deprecated
        printf("<VC-1 Video> (unsupported)\n");
        return false;
    }
    else
    {
        printf("Unknown Video Format \"%s\"\n", pParams->iCodecType);
        return false;
    }

    return true;
}


bool
VideoEncoder::SetParameters(NVEncoderParams *pParams)
{
    assert(pParams != NULL);

    HRESULT hr = S_OK;

    hr = SetParamValue(NVVE_OUT_SIZE,       &(pParams->iOutputSize));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_IN_SIZE,        &(pParams->iInputSize));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_ASPECT_RATIO,   &(pParams->iAspectRatio));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_FIELD_ENC_MODE, &(pParams->Fieldmode));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_P_INTERVAL,     &(pParams->iP_Interval));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_IDR_PERIOD,     &(pParams->iIDR_Period));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_DYNAMIC_GOP,    &(pParams->iDynamicGOP));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_RC_TYPE,        &(pParams->RCType));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_AVG_BITRATE,    &(pParams->iAvgBitrate));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_PEAK_BITRATE,   &(pParams->iPeakBitrate));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_QP_LEVEL_INTRA, &(pParams->iQP_Level_Intra));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_QP_LEVEL_INTER_P,&(pParams->iQP_Level_InterP));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_QP_LEVEL_INTER_B,&(pParams->iQP_Level_InterB));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_FRAME_RATE,     &(pParams->iFrameRate));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_DEBLOCK_MODE,   &(pParams->iDeblockMode));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_PROFILE_LEVEL,  &(pParams->iProfileLevel));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_FORCE_INTRA,    &(pParams->iForceIntra));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_FORCE_IDR,      &(pParams->iForceIDR));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_CLEAR_STAT,     &(pParams->iClearStat));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_SET_DEINTERLACE,&(pParams->DIMode));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    if (pParams->Presets != -1)
    {
        hr = SetParamValue(NVVE_PRESETS,    &(pParams->Presets));

        if (hr!=S_OK)
        {
            return FALSE;
        }
    }

    hr = SetParamValue(NVVE_DISABLE_CABAC,  &(pParams->iDisableCabac));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_CONFIGURE_NALU_FRAMING_TYPE, &(pParams->iNaluFramingType));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    hr = SetParamValue(NVVE_DISABLE_SPS_PPS,&(pParams->iDisableSPSPPS));

    if (hr!=S_OK)
    {
        return FALSE;
    }

    printf("\n");
    return true;
}

int
VideoEncoder::DisplayGPUCaps(int deviceOrdinal, NVEncoderParams *pParams, bool bDisplay)
{
    NVVE_GPUAttributes GPUAttributes = {0};
    HRESULT hr = S_OK;
    int gpuPerformance;

    assert(pParams != NULL);

    GPUAttributes.iGpuOrdinal = deviceOrdinal;
    hr = GetParamValue(NVVE_GET_GPU_ATTRIBUTES,  &GPUAttributes);

    if (hr!=S_OK)
    {
        printf("  >> NVVE_GET_GPU_ATTRIBUTES error! <<\n\n");
    }

    gpuPerformance = GPUAttributes.iClockRate * GPUAttributes.iMultiProcessorCount;
    gpuPerformance = gpuPerformance * _ConvertSMVer2Cores(GPUAttributes.iMajor, GPUAttributes.iMinor);

    size_t totalGlobalMem;
    CUresult error_id = cuDeviceTotalMem(&totalGlobalMem, deviceOrdinal);

    if (error_id != CUDA_SUCCESS)
    {
        printf("cuDeviceTotalMem returned %d\n-> %s\n", (int)error_id, getCudaDrvErrorString(error_id));
        return -1;
    }


    if (bDisplay)
    {
        printf("  GPU Device %d (SM %d.%d) : %s\n", GPUAttributes.iGpuOrdinal,
               GPUAttributes.iMajor, GPUAttributes.iMinor,
               GPUAttributes.cName);
        printf("  Total Memory          = %4.0f MBytes\n" , ceil((float)totalGlobalMem/1048576.0f));
        printf("  GPU Clock             = %4.2f MHz\n"    , (float)GPUAttributes.iClockRate/1000.f);
        printf("  MultiProcessors/Cores = %d MPs (%d Cores)\n", GPUAttributes.iMultiProcessorCount,
               GPUAttributes.iMultiProcessorCount*_ConvertSMVer2Cores(GPUAttributes.iMajor, GPUAttributes.iMinor));
        printf("  Maximum Offload Mode  = ");

        switch (GPUAttributes.MaxGpuOffloadLevel)
        {
            case NVVE_GPU_OFFLOAD_DEFAULT:
                printf("CPU: PEL Processing Only\n");
                break;

            case NVVE_GPU_OFFLOAD_ESTIMATORS:
                printf("GPU: Motion Estimation & Intra Prediction\n");
                break;

            case NVVE_GPU_OFFLOAD_ALL:
                printf("GPU: Full Offload\n");
                break;
        }

        printf("\n");
    }

    pParams->MaxOffloadLevel = GPUAttributes.MaxGpuOffloadLevel;

    return gpuPerformance;
}

int
VideoEncoder::GetGPUCount(NVEncoderParams *pParams, int *gpuPerf, int *bestGPU)
{
    assert(gpuPerf != NULL && bestGPU != NULL && pParams != NULL);

    // Now we can query the GPUs available for encoding
    HRESULT hr = GetParamValue(NVVE_GET_GPU_COUNT, &(pParams->GPU_count));

    if (hr!=S_OK)
    {
        printf("  >> NVVE_GET_GPU_COUNT error ! <<\n\n");
    }

    printf("\n[ Detected %d GPU(s) capable of CUDA Accelerated Video Encoding ]\n\n", pParams->GPU_count);
    int temp = 0;

    for (int deviceCount=0; deviceCount < pParams->GPU_count; deviceCount++)
    {
        temp = DisplayGPUCaps(deviceCount, pParams, !(pParams->force_device));

        if (temp > (*gpuPerf))
        {
            *gpuPerf = temp;
            *bestGPU = deviceCount;
        }
    }

    return (*bestGPU);
}

void
VideoEncoder::SetActiveGPU(NVEncoderParams *pParams, int gpuID)
{
    assert(pParams != NULL);

    printf("  >> Setting Active GPU %d for Video Encoding <<\n", gpuID);
    HRESULT hr = SetParamValue(NVVE_FORCE_GPU_SELECTION, &gpuID);

    if (hr!=S_OK)
    {
        printf("  >> NVVE_FORCE_GPU_SELECTION Error <<\n\n");
    }

    DisplayGPUCaps(gpuID, pParams, true);
}

void
VideoEncoder::SetGPUOffloadLevel(NVEncoderParams *pParams)
{
    assert(pParams != NULL);

    NVVE_GPUOffloadLevel eMaxOffloadLevel = NVVE_GPU_OFFLOAD_DEFAULT;
    HRESULT              hr               = GetParamValue(NVVE_GPU_OFFLOAD_LEVEL_MAX, &eMaxOffloadLevel);

    if (hr!=S_OK)
    {
        printf("  >> NVVE_GPUOFFLOAD_LEVEL_MAX Error <<\n\n");
    }

    if (pParams->GPUOffloadLevel > eMaxOffloadLevel)
    {
        pParams->GPUOffloadLevel = eMaxOffloadLevel;
        printf("  >> Overriding, setting GPU to: ");

        switch (pParams->GPUOffloadLevel)
        {
            case NVVE_GPU_OFFLOAD_DEFAULT:
                printf("Offload Default (CPU: PEL Processing\n)");
                break;

            case NVVE_GPU_OFFLOAD_ESTIMATORS:
                printf("Offload Motion Estimators\n");
                break;

            case NVVE_GPU_OFFLOAD_ALL:
                printf("Offload Full Encode\n)");
                break;
        }
    }

    hr = SetParamValue(NVVE_GPU_OFFLOAD_LEVEL, &(pParams->GPUOffloadLevel));

    if (hr!=S_OK)
    {
        printf("  >> NVVE_GPU_OFFLOAD_LEVEL Error <<\n\n");
    }
}
