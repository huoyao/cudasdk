#define __NV_CUBIN_HANDLE_STORAGE__ static
#include "crt/host_runtime.h"
#include "newdelete.fatbin.c"
typedef Container<int>  _Z9ContainerIiE;
extern void __device_stub__Z12vectorCreatePP9ContainerIiEi( _Z9ContainerIiE **, int);
extern void __device_stub__Z13containerFillPP9ContainerIiE( _Z9ContainerIiE **);
extern void __device_stub__Z16containerConsumePP9ContainerIiEPi( _Z9ContainerIiE **, int *);
extern void __device_stub__Z15containerDeletePP9ContainerIiE( _Z9ContainerIiE **);
extern void __device_stub__Z12placementNewPi(int *);
extern void __device_stub__Z13complexVectorPi(int *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll_28_newdelete_compute_35_cpp1_ii_1e3911aa(void);
#pragma section(".CRT$XCU",read,write)
__declspec(allocate(".CRT$XCU"))static void (*__dummy_static_init__sti____cudaRegisterAll_28_newdelete_compute_35_cpp1_ii_1e3911aa[])(void) = {__sti____cudaRegisterAll_28_newdelete_compute_35_cpp1_ii_1e3911aa};
void __device_stub__Z12vectorCreatePP9ContainerIiEi( _Z9ContainerIiE **__par0, int __par1){__cudaSetupArgSimple(__par0, 0U);__cudaSetupArgSimple(__par1, 4U);__cudaLaunch(((char *)((void ( *)( _Z9ContainerIiE **, int))vectorCreate)));}
#line 34 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
void vectorCreate(  _Z9ContainerIiE **__cuda_0,int __cuda_1)
#line 35 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
{__device_stub__Z12vectorCreatePP9ContainerIiEi( __cuda_0,__cuda_1);
#line 41 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
}
#line 1 "newdelete.compute_20.cudafe1.stub.c"
void __device_stub__Z13containerFillPP9ContainerIiE(  _Z9ContainerIiE **__par0) {  __cudaSetupArgSimple(__par0, 0U); __cudaLaunch(((char *)((void ( *)( _Z9ContainerIiE **))containerFill))); }
#line 52 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
void containerFill(  _Z9ContainerIiE **__cuda_0)
#line 53 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
{__device_stub__Z13containerFillPP9ContainerIiE( __cuda_0);
#line 59 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
}
#line 1 "newdelete.compute_20.cudafe1.stub.c"
void __device_stub__Z16containerConsumePP9ContainerIiEPi(  _Z9ContainerIiE **__par0,  int *__par1) {  __cudaSetupArgSimple(__par0, 0U); __cudaSetupArgSimple(__par1, 4U); __cudaLaunch(((char *)((void ( *)( _Z9ContainerIiE **, int *))containerConsume))); }
#line 62 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
void containerConsume(  _Z9ContainerIiE **__cuda_0,int *__cuda_1)
#line 63 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
{__device_stub__Z16containerConsumePP9ContainerIiEPi( __cuda_0,__cuda_1);
#line 77 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
}
#line 1 "newdelete.compute_20.cudafe1.stub.c"
void __device_stub__Z15containerDeletePP9ContainerIiE(  _Z9ContainerIiE **__par0) {  __cudaSetupArgSimple(__par0, 0U); __cudaLaunch(((char *)((void ( *)( _Z9ContainerIiE **))containerDelete))); }
#line 88 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
void containerDelete(  _Z9ContainerIiE **__cuda_0)
#line 89 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
{__device_stub__Z15containerDeletePP9ContainerIiE( __cuda_0);

}
#line 1 "newdelete.compute_20.cudafe1.stub.c"
void __device_stub__Z12placementNewPi( int *__par0) {  __cudaSetupArgSimple(__par0, 0U); __cudaLaunch(((char *)((void ( *)(int *))placementNew))); }
#line 102 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
void placementNew( int *__cuda_0)
#line 103 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
{__device_stub__Z12placementNewPi( __cuda_0);
#line 137 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
}
#line 1 "newdelete.compute_20.cudafe1.stub.c"
void __device_stub__Z13complexVectorPi( int *__par0) {  __cudaSetupArgSimple(__par0, 0U); __cudaLaunch(((char *)((void ( *)(int *))complexVector))); }
#line 150 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
void complexVector( int *__cuda_0)
#line 151 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
{__device_stub__Z13complexVectorPi( __cuda_0);
#line 191 "d:/bld/rel/gpgpu/toolkit/r5.5/samples/6_Advanced/newdelete/newdelete.cu"
}
#line 1 "newdelete.compute_20.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T2887) {  __nv_dummy_param_ref(__T2887); __cudaRegisterEntry(__T2887, ((void ( *)(int *))complexVector), _Z13complexVectorPi, (-1)); __cudaRegisterEntry(__T2887, ((void ( *)(int *))placementNew), _Z12placementNewPi, (-1)); __cudaRegisterEntry(__T2887, ((void ( *)( _Z9ContainerIiE **))containerDelete), _Z15containerDeletePP9ContainerIiE, (-1)); __cudaRegisterEntry(__T2887, ((void ( *)( _Z9ContainerIiE **, int *))containerConsume), _Z16containerConsumePP9ContainerIiEPi, (-1)); __cudaRegisterEntry(__T2887, ((void ( *)( _Z9ContainerIiE **))containerFill), _Z13containerFillPP9ContainerIiE, (-1)); __cudaRegisterEntry(__T2887, ((void ( *)( _Z9ContainerIiE **, int))vectorCreate), _Z12vectorCreatePP9ContainerIiEi, (-1)); }
static void __sti____cudaRegisterAll_28_newdelete_compute_35_cpp1_ii_1e3911aa(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }
