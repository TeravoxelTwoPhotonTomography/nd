/**
 * \file
 * bitshift each element in a gpu-based array and mask overflow.
 * \todo This only works for contiguous strides.
 */
#include "../core.h"
#include "../ops.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include "macros.h"
TYPEDEFS;

#define ENDL "\n"
#define LOG(...) ndLogError(dst,__VA_ARGS__)
#define TRY(e)   do{if(!(e)) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error; }}while(0)
#define CUTRY(e) do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,cudaGetErrorString(ecode)); goto Error; }}while(0)
#define FAIL     do{LOG("%s(%d) %s()"ENDL "\tExecution should not have reached here."ENDL,__FILE__,__LINE__,__FUNCTION__); goto Error; }while(0)


template<typename T,unsigned BX,unsigned BY,unsigned WORK>
__global__ void __launch_bounds__(BX*BY,1)
leftshift_ip_kernel(T* dst,unsigned w,unsigned h,int b,T mask)
{ const int ox=threadIdx.x+(blockIdx.x*WORK)*BX,
            oy=threadIdx.y+ blockIdx.y      *BY;
  if(oy<h)
  { dst+=ox+oy*(int)w;
    if(blockIdx.x!=(gridDim.x-1))
    {
      #pragma unroll
      for(int i=0;i<WORK;++i) dst[i*BX]=(dst[i*BX]<<b)&mask;
    } else
    { // last block 
      #pragma unroll
      for(int i=0;i<WORK;++i) if(w-ox>i*BX) dst[i*BX]=(dst[i*BX]<<b)&mask;
    }
  } 
}

template<typename T,unsigned BX,unsigned BY,unsigned WORK>
__global__ void __launch_bounds__(BX*BY,1)
rightshift_ip_kernel(T* dst,unsigned w,unsigned h,int b,T mask)
{ const int ox=threadIdx.x+(blockIdx.x*WORK)*BX,
            oy=threadIdx.y+ blockIdx.y      *BY;
  if(oy<h)
  { dst+=ox+oy*(int)w;
    if(blockIdx.x!=(gridDim.x-1))
    {
      #pragma unroll
      for(int i=0;i<WORK;++i) dst[i*BX]=(dst[i*BX]>>b)&mask;
    } else
    { // last block 
      #pragma unroll
      for(int i=0;i<WORK;++i) if(w-ox>i*BX) dst[i*BX]=(dst[i*BX]>>b)&mask;
    }
  } 
}

static unsigned prod(size_t n, size_t *v)
{ unsigned o=1;
  while(n-->0) o*=v[n];
  return o;
}

extern "C" unsigned bitshift_ip_cuda(nd_t dst,int b,int n)
{ unsigned w=ndshape(dst)[0],
           h=prod(ndndim(dst)-1,ndshape(dst)+1);
  const unsigned BX=32,BY=32,WORK=8;
  dim3 blocks((unsigned)ceil(w/(float)(WORK*BX)), (unsigned)ceil(h/(float)BY)),
       threads(BX,BY); // run max threads per block (1024).  Set BX to be 1 warp (32).
  if(ndtype(dst)==nd_f32 || ndtype(dst)==nd_f64)
  { FAIL; //Bitshift of floating types not supported.
  } else
  { if(b<0)
      /// @cond DEFINES
      #define CASE(T) {T mask=((1ULL<<n)-1); rightshift_ip_kernel<T,BX,BY,WORK><<<blocks,threads,0,(cudaStream_t)ndCudaStream(dst)>>>((T*)nddata(dst),w,h,b,mask); } break
           {TYPECASE_INTEGERS(ndtype(dst));}
      #undef CASE 
    else
      #define CASE(T) {T mask=((1ULL<<n)-1); leftshift_ip_kernel<T,BX,BY,WORK><<<blocks,threads,0,(cudaStream_t)ndCudaStream(dst)>>>((T*)nddata(dst),w,h,b,mask); } break
           {TYPECASE_INTEGERS(ndtype(dst));}
      #undef CASE 
    /// @endcond
  }
  CUTRY(cudaGetLastError());
  return 1;
Error:
  return 0;
}