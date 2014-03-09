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

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <string>
#include <types.h>
#include <math.h> // log10


// includes, CUDA
#include <cuda.h>
#include <builtin_types.h>
#include <drvapi_error_string.h>

#include <helper_string.h>
#include <helper_cuda_drvapi.h>
#include <helper_timer.h>

#include "VideoEncoder.h"

using namespace std;

static char *sAppName     = "CUDA H.264 Encoder";
static char *sAppFilename = "cudaEncode.exe";

//////////////////////////////////
// Global Variables Defined

StopWatchInterface *frame_timer  = 0;
StopWatchInterface *global_timer = 0;
unsigned int g_FrameCount = 0;
unsigned int g_fpsCount = 0;      // FPS count for averaging
unsigned int g_fpsLimit = 16;     // FPS limit for sampling timer;

double mse[3] = {0.0};

#define VIDEO_SOURCE_FILE "plush_480p_60fr.yuv"
#define VIDEO_CONFIG_FILE "704x480-h264.cfg"
#define VIDEO_OUTPUT_FILE "plush_480p_60fr.264"

bool ParseInputParams(int argc, char *argv[], NVEncoderParams *pParams);

void computeFPS()
{
    sdkStopTimer(&frame_timer);
    g_FrameCount++;
    g_fpsCount++;

    if (g_fpsCount == g_fpsLimit)
    {
        float ifps = 1.f / (sdkGetAverageTimerValue(&frame_timer) / 1000.f);

        printf("[%s] - [Frame: %04d, %04.1f fps, frame time: %04.2f (ms) ]\n",
               sAppFilename, g_FrameCount, ifps, 1000.f/ifps);

        sdkResetTimer(&frame_timer);
        g_fpsCount = 0;
    }

    sdkStartTimer(&frame_timer);
}

void printHelp()
{
    printf("[ %s Help]\n", sAppName);

    printf("Usage: %s -input=[input.yuv] -cfg=[config.cfg] -out=[output.264] (optional)\n\n", sAppFilename);
    printf(" (optional parameters)\n");
    printf("   [-device=n]     (Specify GPU device (n) to run encode on\n");
    printf("   [-format=xxxx]  (xxxx = UYVY, YUY2, YV12, NV12, IYUV)\n");
    printf("   [-pictype=n]    (1=Interlaced(top), 2=Interlaced (bottom), 3=Frame\n");
    printf("   [-offload=type] (type=\"partial\" (ME) type=\"full\"(Encode)\n");
    printf("   [-vmeminput]   (take input from GPU Device Memory)\n");
}


// Parsing the command line arguments and programming the NVEncoderParameters parameters
bool ParseInputParams(int argc, char *argv[], NVEncoderParams *pParams)
{
    int argcount=0;

    pParams->measure_fps  = 1;
    pParams->measure_psnr = 0;
    pParams->force_device = 0;
    pParams->iForcedGPU   = 0;

    // By default we want to do motion estimation on the GPU
    pParams->GPUOffloadLevel= NVVE_GPU_OFFLOAD_ALL; // NVVE_GPU_OFFLOAD_ESTIMATORS;
    pParams->iSurfaceFormat = (int)YV12;
    pParams->iPictureType   = (int)FRAME_PICTURE;

    printf("[ %s ]\n", sAppName);

    for (argcount=0; argcount  < argc; argcount++)
    {
        printf("  argv[%d] = %s\n", argcount, argv[argcount]);
    }

    printf("\n");

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        return false;
    }

    // By default, we will define the default source and configuration files
    strcpy(pParams->inputFile,  sdkFindFilePath(VIDEO_SOURCE_FILE, argv[0]));
    strcpy(pParams->configFile, sdkFindFilePath(VIDEO_CONFIG_FILE, argv[0]));
    strcpy(pParams->outputFile, VIDEO_OUTPUT_FILE);

    char *filePath = NULL;

    // These are single parameter options that can be passed in directly
    if (argc > 1)
    {
        pParams->iSurfaceFormat = YV12;

        if (checkCmdLineFlag(argc, (const char **)argv, "device"))
        {
            pParams->force_device = 1;
            pParams->iForcedGPU = getCmdLineArgumentInt(argc, (const char **)argv, "device");

            if (pParams->iForcedGPU < 0)
            {
                printf("Command line parameter -device=n must be > 0\n");
            }
        }
    }

    if (argc > 2)
    {
        int filename_length = 0;

        if (checkCmdLineFlag(argc, (const char **)argv, "input"))
        {
            char *file_name = NULL;
            getCmdLineArgumentString(argc, (const char **)argv, "input", &file_name);

            filename_length = (int)strlen(file_name);

            if (!STRCASECMP(&file_name[filename_length-3], "yuv") ||
                !STRCASECMP(&file_name[filename_length-4], "yuy2") ||
                !STRCASECMP(&file_name[filename_length-4], "yv12") ||
                !STRCASECMP(&file_name[filename_length-4], "nv12") ||
                !STRCASECMP(&file_name[filename_length-3], "avi"))
            {
                strcpy(pParams->inputFile, file_name);
            }
            else
            {
                printf("Unsupported filename type <%s>\n", file_name);
            }
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "cfg"))
        {
            char *cfg_name = NULL;
            getCmdLineArgumentString(argc, (const char **)argv, "cfg", &cfg_name);

            filename_length = (int)strlen(cfg_name);

            if (!STRCASECMP(&cfg_name[filename_length-3], "cfg"))
            {
                strcpy(pParams->configFile, sdkFindFilePath(cfg_name, argv[0]));
            }
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "out"))
        {
            char *output_name;
            getCmdLineArgumentString(argc, (const char **)argv, "out", &output_name);

            filename_length = (int)strlen(output_name);

            if (!STRCASECMP(&output_name[filename_length-3], "264") ||
                !STRCASECMP(&output_name[filename_length-4], "h264"))
            {
                strcpy(pParams->outputFile, output_name);
            }
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "format"))
        {
            char *format = NULL;
            getCmdLineArgumentString(argc, (const char **)argv, "format", &format);

            printf("YUV Surface Format: ");

            if (!STRCASECMP(format, "UYVY"))
            {
                pParams->iSurfaceFormat = (int)UYVY;
            }
            else if (!STRCASECMP(format, "YUY2"))
            {
                pParams->iSurfaceFormat = (int)YUY2;
            }
            else if (!STRCASECMP(format, "YV12"))
            {
                pParams->iSurfaceFormat = (int)YV12;
            }
            else if (!STRCASECMP(format, "NV12"))
            {
                pParams->iSurfaceFormat = (int)NV12;
            }
            else if (!STRCASECMP(format, "IYUV"))
            {
                pParams->iSurfaceFormat = (int)IYUV;
            }

            printf("%s\n", sSurfaceFormat[pParams->iSurfaceFormat]);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "pictype"))
        {
            printf("Frame Type: ");
            pParams->iPictureType = getCmdLineArgumentInt(argc, (const char **)argv, "pictype");

            if (pParams->iPictureType >= 1 && pParams->iPictureType <= 3)
            {
                printf("%s\n", sPictureStructure[pParams->iPictureType]);
            }
            else
            {
                printf(" %d is invalid!\n", pParams->iPictureType);
            }
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "offload"))
        {
            char *mode = NULL;
            getCmdLineArgumentString(argc, (const char **)argv, "offload", &mode);

            printf("GPU Offload Mode: ");

            if (!STRCASECMP(mode, "partial"))
            {
                pParams->GPUOffloadLevel = NVVE_GPU_OFFLOAD_ESTIMATORS;
                printf("<%s>\n", sGPUOffloadLevel[1]);
            }
            else if (!STRCASECMP(mode, "full"))
            {
                pParams->GPUOffloadLevel = NVVE_GPU_OFFLOAD_ALL;
                printf("<%s>\n", sGPUOffloadLevel[2]);
            }
            else
            {
                pParams->GPUOffloadLevel = NVVE_GPU_OFFLOAD_ESTIMATORS ;
                printf("<%s>\n", sGPUOffloadLevel[1]);
            }
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "vmeminput"))
        {
            pParams->iUseDeviceMem = 1;
            pParams->iUseDeviceMem = getCmdLineArgumentInt(argc, (const char **)argv, "vmeminput");
        }
    }
    else
    {
        // This is demo mode, we will print out the help, and run the encode
        printf("\n[Demonstration Mode]\n\n");
        pParams->iSurfaceFormat = YV12;
    }

    if (!strlen(pParams->configFile))
    {
        printf("\n *.cfg config file is required to use the encoder\n");
        return false;
    }

    if (!strlen(pParams->inputFile))
    {
        printf("\n *.yuv input file is required to use the encoder\n");
        return false;
    }

    if (!strlen(pParams->outputFile))
    {
        printf("\n *.264 output file is required to use the encoder\n");
        return false;
    }

    return true;
}


// NVCUVENC callback function to signal the start of bitstream that is to be encoded
static unsigned char *_stdcall HandleAcquireBitStream(int *pBufferSize, void *pUserData)
{
    VideoEncoder *pCudaEncoder;

    if (pUserData)
    {
        pCudaEncoder = (VideoEncoder *)pUserData;
    }
    else
    {
        printf(">> VideoEncoder structure is invalid!\n");
    }

    *pBufferSize = 1024*1024;
    return pCudaEncoder->GetCharBuf();
}

//NVCUVENC callback function to signal that the encoded bitstream is ready to be written to file
static void _stdcall HandleReleaseBitStream(int nBytesInBuffer, unsigned char *cb,void *pUserData)
{
    VideoEncoder *pCudaEncoder;

    if (pUserData)
    {
        pCudaEncoder = (VideoEncoder *)pUserData;
    }
    else
    {
        printf(">> VideoEncoder structure is invalid!\n");
        return;
    }

    if (pCudaEncoder && pCudaEncoder->fileOut())
    {
        fwrite(cb,1,nBytesInBuffer,pCudaEncoder->fileOut());
    }

    return;
}

//NVCUVENC callback function to signal that the encoding operation on the frame has started
static void _stdcall HandleOnBeginFrame(const NVVE_BeginFrameInfo *pbfi, void *pUserData)
{
    return;
}

//NVCUVENC callback function signals that the encoding operation on the frame has finished
static void _stdcall HandleOnEndFrame(const NVVE_EndFrameInfo *pefi, void *pUserData)
{
    VideoEncoder *pCudaEncoder;

    if (pUserData)
    {
        pCudaEncoder = (VideoEncoder *)pUserData;
    }
    else
    {
        printf(">> VideoEncoder structure is invalid!\n");
        return;
    }

    pCudaEncoder->frameSummation(-(pefi->nFrameNumber));

    if (pCudaEncoder->IsLastFrameSent())
    {
        // Check to see if the last frame has been sent
        if (pCudaEncoder->getFrameSum() == 0)
        {
            printf(">> Encoder has finished encoding last frame\n<< n");
        }
    }
    else
    {
#ifdef _DEBUG
        //        printf("HandleOnEndFrame (%d), FrameCount (%d), FrameSummation (%d)\n",
        //          pefi->nFrameNumber, pCudaEncoder->frameCount(),
        //          pCudaEncoder->getFrameSum());
#endif
    }

    return;
}


// This is our main application code
int main(int argc, char *argv[])
{
    HRESULT hr = S_OK;
    int retvalue = -1;

    printf("Starting cudaEncode...\n");

    // NVCUVENC data structures and wrapper class
    VideoEncoder        *pCudaEncoder  = NULL;
    NVEncoderParams     sEncoderParams = {0};
    NVVE_CallbackParams sCBParams      = {0};

    // CUDA resources needed (for CUDA Encoder interop with a previously created CUDA Context, and accepting GPU video memory)
    CUcontext      cuContext;
    CUdevice       cuDevice;
    CUvideoctxlock cuCtxLock      = 0;
    CUdeviceptr    dptrVideoFrame = 0;

    unsigned int Pitch, Height, WidthInBytes, ElementSizeBytes;

    void *pData = NULL;

    // First we parse the input file (based on the command line parameters)
    // Set the input/output filenmaes
    if (!ParseInputParams(argc, argv, &sEncoderParams))
    {
        printHelp();
        exit(EXIT_SUCCESS);
    }

    // Create the NVCUVENC wrapper class for handling encoding
    pCudaEncoder = new VideoEncoder(&sEncoderParams);
    pCudaEncoder->InitEncoder(&sEncoderParams);
    pCudaEncoder->SetEncodeParameters(&sEncoderParams);

    // Create the timer for frame time measurement
    sdkCreateTimer(&frame_timer);
    sdkResetTimer(&frame_timer) ;

    sdkCreateTimer(&global_timer);
    sdkResetTimer(&global_timer) ;

    // This is for GPU device memory input, and support for interop with another CUDA context
    // The NVIDIA CUDA Encoder will use this CUDA context to be able to pass in shared device memory
    if (sEncoderParams.iUseDeviceMem)
    {
        HRESULT hr = S_OK;
        printf(">> Using Device Memory for Video Input to CUDA Encoder << \n");

        // Create the CUDA context
        checkCudaErrors(cuInit(0));
        checkCudaErrors(cuDeviceGet(&cuDevice, sEncoderParams.iForcedGPU));
        checkCudaErrors(cuCtxCreate(&cuContext, CU_CTX_BLOCKING_SYNC, cuDevice));

        // Allocate the CUDA memory Pitched Surface
        if (sEncoderParams.iSurfaceFormat == UYVY ||
            sEncoderParams.iSurfaceFormat == YUY2)
        {
            WidthInBytes     =(sEncoderParams.iInputSize[0] * sSurfaceFormat[sEncoderParams.iSurfaceFormat].bpp) >> 3; // Width
            Height           = sEncoderParams.iInputSize[1];
        }
        else
        {
            WidthInBytes     = sEncoderParams.iInputSize[0]; // Width
            Height           = (unsigned int)(sEncoderParams.iInputSize[1] * sSurfaceFormat[sEncoderParams.iSurfaceFormat].bpp) >> 3;
        }

        ElementSizeBytes = 16;
#if (CUDA_FORCE_API_VERSION == 3010)
        checkCudaErrors(cuMemAllocPitch(&dptrVideoFrame, &Pitch, WidthInBytes, Height, ElementSizeBytes));
#else
        checkCudaErrors(cuMemAllocPitch(&dptrVideoFrame, (size_t *)&Pitch, WidthInBytes, Height, ElementSizeBytes));
#endif

        sEncoderParams.nDeviceMemPitch = Pitch; // Copy the Device Memory Pitch (we'll need this for later if we use device memory)

        // Pop the CUDA context from the stack (this will make the CUDA context current)
        // This is needed in order to inherit the CUDA contexts created outside of the CUDA H.264 Encoder
        // CUDA H.264 Encoder will just inherit the available CUDA context
        CUcontext cuContextCurr;
        checkCudaErrors(cuCtxPopCurrent(&cuContextCurr));

        // Create the Video Context Lock (used for synchronization)
        checkCudaErrors(cuvidCtxLockCreate(&cuCtxLock, cuContext));

        // If we are using GPU Device Memory with NVCUVENC, it is necessary to create a
        // CUDA Context with a Context Lock cuvidCtxLock.  The Context Lock needs to be passed to NVCUVENC
        {
            hr = pCudaEncoder->SetParamValue(NVVE_DEVICE_MEMORY_INPUT, &(sEncoderParams.iUseDeviceMem));

            if (FAILED(hr))
            {
                printf("NVVE_DEVICE_MEMORY_INPUT failed\n");
            }

            hr = pCudaEncoder->SetParamValue(NVVE_DEVICE_CTX_LOCK    , &cuCtxLock);

            if (FAILED(hr))
            {
                printf("NVVE_DEVICE_CTX_LOCK failed\n");
            }
        }
    }

    // Now provide the callback functions to CUDA H.264 Encoder
    {
        memset(&sCBParams,0,sizeof(NVVE_CallbackParams));
        sCBParams.pfnacquirebitstream = HandleAcquireBitStream;
        sCBParams.pfnonbeginframe     = HandleOnBeginFrame;
        sCBParams.pfnonendframe       = HandleOnEndFrame;
        sCBParams.pfnreleasebitstream = HandleReleaseBitStream;

        pCudaEncoder->SetCBFunctions(&sCBParams, (void *)pCudaEncoder);
    }

    // Now we must create the HW Encoder device
    pCudaEncoder->CreateHWEncoder(&sEncoderParams);

    // CPU Timers needed for performance
    {
        sdkStartTimer(&global_timer);
        sdkResetTimer(&global_timer);

        sdkStartTimer(&frame_timer);
        sdkResetTimer(&frame_timer);
    }

    // This is the loop needed for reading in the input source, packaging up the frame structure to be
    // sent to the CUDA H.264 Encoder
    while (!feof(pCudaEncoder->fileIn()))
    {
        NVVE_EncodeFrameParams      efparams;
        efparams.Height           = sEncoderParams.iOutputSize[1];
        efparams.Width            = sEncoderParams.iOutputSize[0];
        efparams.Pitch            = (sEncoderParams.nDeviceMemPitch ? sEncoderParams.nDeviceMemPitch : sEncoderParams.iOutputSize[0]);
        efparams.PictureStruc     = (NVVE_PicStruct)sEncoderParams.iPictureType;
        efparams.SurfFmt          = (NVVE_SurfaceFormat)sEncoderParams.iSurfaceFormat;
        efparams.progressiveFrame = (sEncoderParams.iSurfaceFormat == 3) ? 1 : 0;
        efparams.repeatFirstField = 0;
        efparams.topfieldfirst    = (sEncoderParams.iSurfaceFormat == 1) ? 1 : 0;

        // see VideoEncoder.cpp
        size_t bytes_read = pCudaEncoder->ReadNextFrame();

        if (bytes_read != -1)
        {
            efparams.picBuf = (unsigned char *)pCudaEncoder->GetVideoFrame(); ;//get the yuv buffer pointer from file
        }
        else
        {
            printf("Error, Invalid Frame read\n");
            exit(EXIT_SUCCESS);
        }

        // Once we have reached the EOF, we know this is the last frame
        // Send this flag to the H.264 Encoder so it knows to properly flush out the file
        if (!feof(pCudaEncoder->fileIn()))
        {
            efparams.bLast = false;
        }
        else
        {
            efparams.bLast = true;
        }

        // If dptrVideoFrame is NULL, then we assume that frames come from system memory, otherwise it comes from GPU memory
        // VideoEncoder.cpp, EncodeFrame() will automatically copy it to GPU Device memory, if GPU device input is specified
        if (pCudaEncoder->EncodeFrame(efparams, dptrVideoFrame, cuCtxLock) == false)
        {
            printf("\nEncodeFrame() failed to encode frame\n");
        }

        computeFPS();
    }

    pCudaEncoder->Stop();

    sdkStopTimer(&global_timer);

    retvalue = 0;

    //clean up stuff, release resources etc
    delete pCudaEncoder;
    pCudaEncoder = NULL;

    // free up resources (device_memory video frame, context lock, CUDA context)
    if (sEncoderParams.iUseDeviceMem)
    {
        checkCudaErrors(cuvidCtxLock(cuCtxLock, 0));
        checkCudaErrors(cuMemFree(dptrVideoFrame));
        checkCudaErrors(cuvidCtxUnlock(cuCtxLock, 0));

        checkCudaErrors(cuvidCtxLockDestroy(cuCtxLock));
        checkCudaErrors(cuCtxDestroy(cuContext));
    }

    printf("\n> %s encoded %s, return value = %d\n", sAppFilename, (retvalue ? "with errors " : "OK"), retvalue);
    exit(!retvalue ? EXIT_SUCCESS : EXIT_FAILURE);
}
