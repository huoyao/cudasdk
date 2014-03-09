Sample: CUDA Video Decoder D3D9 API
Minimum spec: SM 1.1

This sample demonstrates how to efficiently use the CUDA Video Decoder API to decode MPEG-2, VC-1, or H.264 sources.  YUV to RGB conversion of video is accomplished with CUDA kernel.  The output result is rendered to a D3D9 surface.  The decoded video is not displayed on the screen, but with -displayvideo at the command line parameter, the video output can be seen.  Requires a Direct3D capable device and Compute Capability 1.1 or higher.

Key concepts:
