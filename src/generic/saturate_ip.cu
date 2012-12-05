/**
 * \file
 * Fill a gpu-based array with a constant value.
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

typedef union val_t_ { int d; unsigned u; unsigned long long llu; long long lld; double f;} val_t; ///< Generic value type for passing parameters.
#define VAL_u8(v)  (v.u)
#define VAL_u16(v) (v.u)
#define VAL_u32(v) (v.u)
#define VAL_u64(v) (v.llu)
#define VAL_i8(v)  (v.d)
#define VAL_i16(v) (v.d)
#define VAL_i32(v) (v.d)
#define VAL_i64(v) (v.lld)
#define VAL_f32(v) (v.f)
#define VAL_f64(v) (v.f)
#define VAL(v,type) VAL_##type(v)

template<typename T> __device__ const T& clamp(const T& v, const T& mn, const T& mx)
{ if(v<mn)       return mn;
  else if (v>mx) return mx;
  else           return v;
}

/* Could be more efficient?  First id which threads in a block need to be set.  Then coallesce and write. */
template<typename T,unsigned BX,unsigned BY,unsigned WORK>
__global__ void __launch_bounds__(BX*BY,1)
saturate_ip_kernel(T* dst,unsigned w,unsigned h,T mn, T mx)
{ const int ox=threadIdx.x+(blockIdx.x*WORK)*BX,
            oy=threadIdx.y+ blockIdx.y      *BY;
  if(oy<h)
  { dst+=ox+oy*(int)w;
    if(blockIdx.x!=(gridDim.x-1))
    {
      #pragma unroll
      for(int i=0;i<WORK;++i) dst[i*BX]=clamp(dst[i*BX],mn,mx);
    } else
    { // last block - bounds check
      #pragma unroll
      for(int i=0;i<WORK;++i) if(w-ox>i*BX) dst[i*BX]=clamp(dst[i*BX],mn,mx);
    }
  } 
}

static unsigned prod(size_t n, size_t *v)
{ size_t o=1;
  while(n-->0) o*=v[n];
  return (unsigned)o;
}

extern "C" unsigned saturate_ip_cuda(nd_t dst,val_t mn, val_t mx)
{ unsigned w=ndshape(dst)[0],
           h=prod(ndndim(dst)-1,ndshape(dst)+1);
  const unsigned BX=32,BY=32,WORK=8;
  dim3 blocks((unsigned)ceil(w/(float)(WORK*BX)), (unsigned)ceil(h/(float)BY)),
       threads(BX,BY); // run max threads per block (1024).  Set BX to be 1 warp (32).
  /// @cond DEFINES
  #define CASE(T) saturate_ip_kernel<T,BX,BY,WORK><<<blocks,threads,0,(cudaStream_t)ndCudaStream(dst)>>>((T*)nddata(dst),w,h,VAL(mn,T),VAL(mx,T)); break
       {TYPECASE(ndtype(dst));}
  #undef CASE
  /// @endcond
  CUTRY(cudaGetLastError());
  return 1;
Error:
  return 0;
}