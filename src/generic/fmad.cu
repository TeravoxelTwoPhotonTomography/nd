/**
 * \file
 * z = a.*x+b where a,x, and b are gpu based arrays and .* is the matlab-style 
 * multiplication operation.
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
TYPEDEFS; ///< Typedef aliases for basic types.  See generic/macros.h

#define ENDL "\n"
#define LOG(...) ndLogError(z,__VA_ARGS__)
#define REPORT(msg1,msg2) LOG("%s(%d): %s()"ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,msg1,msg2)
#define TRY(e)   do{if(!(e)) {REPORT("Expression evaluated as failure.",#e); goto Error; }}while(0)
#define CUTRY(e) do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {REPORT(#e,cudaGetErrorString(ecode)); goto Error; }}while(0)
#define FAIL     do{REPORT("Execution should not have reached here.","Failing."); goto Error; } while(0)

#ifndef restrict
#define restrict __restrict__
#endif

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
#define min_i32  INT_MIN
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

template<typename T> inline __device__ T saturate(float v);
#define DECL(T) \
  template<> inline T saturate<T>(float v) \
  { const float mn=min_(v,max_##T); \
    return max_(min_##T,mn); \
  }
DECL(u8);   DECL(u16);   DECL(u32);   DECL(u64);
DECL(i8);   DECL(i16);   DECL(i32);   DECL(i64);
DECL(f32);  DECL(f64);
#undef DECL

typedef struct shape_t_
{ u8                 ndim;      ///< The number of dimensions
  u32                nelem;     ///< The total number of elements
  size_t   *restrict shape;     ///< Buffer of length ndim,  ordered [w,h,d,...].  Always agrees with stride.  Maintained for convenience.
} shape_t;

template<typename T>
struct vol_t
{ size_t   *restrict strides;   ///< Buffer of length ndim+1, strides[i] is the number of bytes layed out between unit steps along dimension i
  T        *restrict data;
};

static shape_t make_shape(const nd_t a)
{ shape_t out = 
  { (u8)     ndndim(a),
    (u32)    ndnelem(a),
    (size_t*)ndCudaShape(a)
  };
  return out;
}

template<typename T>
static vol_t<T> make_vol(const nd_t a)
{ vol_t<T> out = 
  { (size_t*)ndCudaStrides(a),
    (T*)nddata(a)
  };
  return out;
}

#if 0
#define __launch_bounds__ (...)
#endif

inline __device__ unsigned prod(dim3 a)            {return a.x*a.y*a.z;}
inline __device__ unsigned stride(uint3 a, dim3 b) {return a.x+b.x*(a.y+b.y*a.z);}
inline __device__ unsigned sum(uint3 a)            {return a.x+a.y+a.z;}

// i = s[0]*(r[0]+s[1]*(r[1]+s[2]*(r[2]+...)))
// r[0]=[i/sh[0]]%r[1]
// r[1]=[i/(sh[0]sh[1])]%sh[2]
// ...
// 

// [ ] Accomidate possible overlap between src/dst arrays.
// [x] Do work elements per thread
// [ ] requires shape[0] aligned to WORK
// [ ] bad memory access pattern - want to do BX*WORK (Bx=32) loads, and then distribute over BY elements.
//     - then we'd require shape[0] aligned to BX*WORK (or do bounds checking)
template<typename TDST,typename TSRC,unsigned BX,unsigned BY,unsigned WORK>
__global__ void __launch_bounds__(BX*BY,1)
fmad_kernel(vol_t<TDST> z, vol_t<TSRC> a, vol_t<TSRC> x, vol_t<TSRC> b, shape_t shape)
{ unsigned i = WORK*(sum(threadIdx)+stride(blockIdx,gridDim)*prod(blockDim));
#if 1
  if(i<shape.nelem)
  { unsigned st=1;
    TDST     *zz=z.data;
    TSRC     *aa=a.data,
             *xx=x.data,
             *bb=b.data;
    for(u8 dim=0;dim<shape.ndim;++dim)
    { unsigned r=(i/st)%shape.shape[dim];
      st*=shape.shape[dim];
      zz+=r*z.strides[dim];
      aa+=r*a.strides[dim];
      xx+=r*x.strides[dim];
      bb+=r*b.strides[dim];
      if(threadIdx.x==7 && threadIdx.y==0)
      { printf("i: %5u\tshape[dim]: %5u\tst: %5u\tr: %5u\n",i,shape.shape[dim],st,r);
      }
    }
    if(i<(((int)shape.nelem)-BX*WORK))
    { 
      #pragma unroll
      for(unsigned j=0;j<WORK;++j)
        zz[j*BX]=aa[j*BX]*xx[j*BX]+bb[j*BX];
    } else
    {
      for(unsigned j=0;j<(shape.nelem-i)/BX;++j)
        zz[j*BX]=aa[j*BX]*xx[j*BX]+bb[j*BX];
    }
  }
#endif
}




// Treat this as a 1d problem, each thread does WORK elements.
// [ ] FIXME - use shape properly
extern "C" unsigned fmad_cuda(nd_t z,nd_t a,nd_t x,nd_t b,size_t ndim,size_t *shape)
{ unsigned n=ndnelem(z);
  const unsigned BX=32,BY=32,WORK=1;
  dim3 blocks((unsigned)ceil(n/(float)(WORK*BX*BY)),1),
       threads(BX,BY);
  /// @cond DEFINES
  #define V(a,T) make_vol<T>(a)
  #define S(a)   make_shape(a)
  #define CASE2(TDST,TSRC) fmad_kernel<TDST,TSRC,BX,BY,WORK><<<blocks,threads,0,(cudaStream_t)ndCudaStream(z)>>>(V(z,TDST),V(a,TSRC),V(x,TSRC),V(b,TSRC),S(z));break
  #define CASE(TSRC)       TYPECASE2(ndtype(z),TSRC); break
       {TYPECASE(ndtype(x));}
  #undef CASE
  #undef CASE2
  #undef V
  #undef S
  /// @endcond
  CUTRY(cudaGetLastError());
  return 1;
Error:
  return 0;
}