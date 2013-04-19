/**
 * \file
 * Scalar multiply and add for all elements in a gpu based buffer.
 * \todo This only works for contiguous strides.
 */
#include "../core.h"
#include "../ops.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <float.h>
#include <algorithm>
#include "macros.h"
TYPEDEFS;

#define ENDL "\n"
#define LOG(...) ndLogError(dst,__VA_ARGS__)
#define TRY(e)   do{if(!(e)) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error; }}while(0)
#define CUTRY(e) do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,cudaGetErrorString(ecode)); goto Error; }}while(0)
#define FAIL     LOG("%s(%d) %s()"ENDL "\tExecution should not have reached here."ENDL,__FILE__,__LINE__,__FUNCTION__); goto Error

#define max_(a,b) (((a)<(b))?(b):(a))
#define min_(a,b) (((a)<(b))?(a):(b))

// std::numeric_limits<T>::min() is a host function; can't be used on device.
// template static const init is not allowed in cuda (for the device code),
// so this:
#define min_u8   0
#define min_u16  0
#define min_u32  0
#define min_u64  0
#define min_i8   CHAR_MIN
#define min_i16  SHRT_MIN
#define min_i32  LONG_MIN
#define min_i64  LLONG_MIN
#define min_f32 (-FLT_MAX)
#define min_f64 (-DBL_MAX)
#define max_u8   UCHAR_MAX
#define max_u16  USHRT_MAX
#define max_u32  ULONG_MAX
#define max_u64  ULLONG_MAX
#define max_i8   CHAR_MAX
#define max_i16  SHRT_MAX
#define max_i32  LONG_MAX
#define max_i64  LLONG_MAX
#define max_f32  FLT_MAX
#define max_f64  DBL_MAX

template<typename T> __device__ T saturate(float v);
#define DECL(T) \
  template<> T saturate<T>(float v) \
  { const float mn=min_(v,max_##T); \
    return max_(min_##T,mn); \
  }
DECL(u8);   DECL(u16);   DECL(u32);   DECL(u64);
DECL(i8);   DECL(i16);   DECL(i32);   DECL(i64);
DECL(f32);  DECL(f64);
#undef DECL

template<typename T,unsigned BX,unsigned WORK>
__global__ void __launch_bounds__(BX,1)
fmad_scalar_ip_kernel(T* dst,unsigned n,float m,float b)
{ const int ox=threadIdx.x+(blockIdx.x*WORK)*BX;
  dst+=ox;
  if(blockIdx.x!=(gridDim.x-1))
  {
    #pragma unroll
    for(int i=0;i<WORK;++i) dst[i*BX]=saturate<T>(m*dst[i*BX]+b);
  } else
  { // last block 
    #pragma unroll
    for(int i=0;i<WORK;++i) if(n-ox>i*BX) dst[i*BX]=saturate<T>(m*dst[i*BX]+b);
  } 
}

extern "C" unsigned fmad_scalar_ip_cuda(nd_t dst,float m, float b)
{ unsigned n=ndnelem(dst);
  const unsigned BX=1024,WORK=8;
  unsigned blocks=(unsigned)ceil(n/(float)(WORK*BX));
  /// @cond DEFINES
  #define CASE(T) fmad_scalar_ip_kernel<T,BX,WORK><<<blocks,BX,0,(cudaStream_t)ndCudaStream(dst)>>>((T*)nddata(dst),n,m,b); break
       {TYPECASE(ndtype(dst));}
  #undef CASE
  /// @endcond
  CUTRY(cudaGetLastError());
  return 1;
Error:
  return 0;
}