/**
 * \file
 * nd affine transform on the GPU with CUDA.
 *
 * \todo Specialized 2d and 3d (maybe 4d) implementations should be faster.  Should do these since they are much more common. 
 * \todo See if using a thread for more than 1 pixel at a time helps
 * \todo reduce register usage (50/thread at the moment!)
 */
#include "../core.h"
#include "../ops.h"

#include "stdio.h"
#include <stdint.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"


#include "generic/macros.h"
TYPEDEFS;

/// @cond DEFINES
#define MAXDIMS          8  // should be sizeof uchar
#define WARPS_PER_BLOCK  4
#define BLOCKSIZE       (32*WARPS_PER_BLOCK) // threads per block

//#define DEBUG_OUTPUT
#ifdef DEBUG_OUTPUT
#define DBG(...) printf(__VA_ARGS__)
#else
#define DBG(...)
#endif

#define ENDL "\n"
#define LOG(...) ndLogError(dst_,__VA_ARGS__)
#define TRY(e) do{if(!(e)) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error; }}while(0)
#define CUTRY(e) do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,cudaGetErrorString(ecode)); goto Error; }}while(0)
#define FAIL     LOG("%s(%d) %s()"ENDL "\tExecution should not have reached here."ENDL,__FILE__,__LINE__,__FUNCTION__); goto Error

#ifndef restrict
#define restrict __restrict__
#endif

// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif
/// @cond DEFINES

typedef uint8_t      u8;
typedef uint32_t     u32;
typedef unsigned int uint;

typedef struct arg_t_
{ u8                 ndim;      ///< The number of dimensions
  u32                nelem;     ///< The total number of elements
  size_t   *restrict shape;     ///< Buffer of length ndim,  ordered [w,h,d,...].  Always agrees with stride.  Maintained for convenience.
  size_t   *restrict strides;   ///< Buffer of length ndim+1, strides[i] is the number of bytes layed out between unit steps along dimension i  
  void     *restrict data;      ///< A poitner to the data.
} arg_t;

inline __device__ float clamp(float f, float a, float b)          {return fmaxf(a, fminf(f, b));}
template<class T> inline __device__ T saturate(float f);
template<> inline __device__ uint8_t  saturate<uint8_t>(float f)  {return clamp(f,0,UCHAR_MAX);}
template<> inline __device__ uint16_t saturate<uint16_t>(float f) {return clamp(f,0,USHRT_MAX);}
template<> inline __device__ uint32_t saturate<uint32_t>(float f) {return clamp(f,0,ULONG_MAX);}
template<> inline __device__ uint64_t saturate<uint64_t>(float f) {return clamp(f,0,ULLONG_MAX);} // FIXME - will overflow float type
template<> inline __device__  int8_t  saturate< int8_t> (float f) {return clamp(f,CHAR_MIN,CHAR_MAX);}
template<> inline __device__  int16_t saturate< int16_t>(float f) {return clamp(f,SHRT_MIN,SHRT_MAX);}
template<> inline __device__  int32_t saturate< int32_t>(float f) {return clamp(f,LONG_MIN,LONG_MAX);}
template<> inline __device__  int64_t saturate< int64_t>(float f) {return clamp(f,LLONG_MIN,LLONG_MAX);} // FIXME - will overflow float type
template<> inline __device__  float   saturate<float>(float f)    {return f;}
template<> inline __device__  double  saturate<double>(float f)   {return f;}

#if 0
/**
 * nD linear interpolation for maximum intensity composting.
 *
 * The boundary handling used here is designed for maximum intensity composting.
 * A constant (determined by \a param->boundary_value) is returned for 
 * out-of-bounds samples.
 * Samples stradling the border are handled as a special case.
 *
 * OUT OF USE
 * Keeping it here because it was interesting. Might need it in the future?  Slow at the moment.
 */
template<class Tsrc,class Tdst>
inline __device__ Tdst sample(arg_t &src,const float *restrict const r,const nd_affine_params_t*const param)
{ uchar2 bounds=inbounds(src.ndim,src.shape,r); // bit i set if inbounds on dim i  
  // clamp to boundary value for out-of-bounds
  if(!bounds.x && !bounds.y)
    return param->boundary_value;
  
  // compute offset to top left ish corner of lattice unit
  u32 idx=0;
  for(u8 i=0;i<src.ndim;++i) 
    idx+=src.strides[i]*floorf(r[i]);

  // iterate over each corner of hypercube
  float v(0.0f);
  for(u8 i=0;i<((1<<src.ndim)-1);++i)              // bits of i select left or right sample on each dimension
  { uchar2 o=make_uchar2(~i&~bounds.x,i&~bounds.y);// don't need to mask high bits of i
    float w=1.0f;
    int offset=0; // offset so corner clamps to edge
    for(u8 idim=0;idim<src.ndim;++idim)            // loop for dot-products w bit vector
    { const size_t s=src.strides[idim];
      const float  a=fpartf(r[idim]),
                   b=1.0f-a;
#define BIT(bs_,i_) (((bs_)&(1<<(i_)))!=0)
      offset+=BIT(o.x,idim)*s          // clamp corner (top left ish)
             -BIT(o.y,idim)*s          // clamp corner (bot right ish)
             +BIT(i,idim)  *s;         // normal corner offset
      w*=BIT(i,idim)*a+BIT(~i,idim)*b; // weight for corner is a product of lerp weights for each dimension
#undef BIT
    }
    v+=w*((Tsrc*)src.data)[idx+offset];             // weighted value for corner
  }
  return saturate<Tdst>(v);
}
#endif

#define max(a,b) ((a)>(b))?(a):(b)

inline __device__ unsigned prod(dim3 a)            {return a.x*a.y*a.z;}
inline __device__ unsigned stride(uint3 a, dim3 b) {return a.x+b.x*(a.y+b.y*a.z);}
inline __device__ unsigned sum(uint3 a)            {return a.x+a.y+a.z;}


/** \todo Respect strides.  Currently assumes strides reflect shape. */
template<typename Tsrc,typename Tdst> 
__global__ void 
__launch_bounds__(BLOCKSIZE,1) /*max threads,min blocks*/
  affine_kernel(arg_t dst, arg_t src, const float *transform, const nd_affine_params_t param)
{ 
  Tdst     o,v;
  //unsigned idst=threadIdx.x+blockIdx.x*blockDim.x;
  unsigned idst = sum(threadIdx)+stride(blockIdx,gridDim)*prod(blockDim);
#if 0
  if(blockIdx.x==0 && threadIdx.x==2)
    printf("ksize src:%d dst:%d\n",(int)sizeof(*ibuf),(int)sizeof(*obuf));
#endif
  if(idst<dst.nelem)
  {
    /////
    unsigned isrc=0;
    u8 oob=0;
#if 1 // 30 ms without this block, 200 ms with (64x64x64x64)
    for(u8 r=0;r<src.ndim;++r)
    { float fcoord=0.0f;
      unsigned i=idst,o=(dst.ndim+1)*r;
      for(u8 c=0;c<dst.ndim;++c)
      { fcoord+=(i%dst.shape[c])*transform[o+c];
        i/=dst.shape[c];
      }
      fcoord+=transform[o+dst.ndim];
      int coord = floor(fcoord);
      if(coord<0 || src.shape[r]<=coord)
      { oob=1;
        break;
      }
      isrc+=src.strides[r]*coord;
    }
#endif
    v=(oob)?param.boundary_value:saturate<Tdst>(*(Tsrc*)((u8*)src.data+isrc));
    /////
    o=((Tdst*)dst.data)[idst];
    ((Tdst*)dst.data)[idst]=max(o,v);
  }
}

static arg_t make_arg(const nd_t a)
{ arg_t out = 
  { (u8)     ndndim(a),
    (u32)    ndnelem(a),
    (size_t*)ndCudaShape(a),
    (size_t*)ndCudaStrides(a),
    nddata(a)
  };
  return out;
}

//
// === Interface ===
//
#include <math.h>
static unsigned nextdim(unsigned n, unsigned limit, unsigned *rem)
{ unsigned v=limit,c=limit,low=n/limit,argmin=0,min=limit;
  *rem=0;  
  if(n<limit) return n;
  for(c=low+1;c<limit&&v>0;c++)
  { v=c*ceil(n/(float)c)-n;
    if(v<min)
    { min=v;
      argmin=c;
    }
  }
  *rem= (min!=0);
  return argmin;
}


/**
 * Assume the ndkind() of \a src_ and \a dst_ have already been checked.
 */
extern "C" unsigned ndaffine_cuda(nd_t dst_, const nd_t src_, const float *transform, const nd_affine_params_t *param)
{ arg_t dst=make_arg(dst_),
        src=make_arg(src_);
  unsigned r,blocks=(unsigned)ceil(dst.nelem/float(BLOCKSIZE)),
           tpb   =BLOCKSIZE;  
  //unsigned b=blocks;
  struct cudaDeviceProp prop;
  dim3 grid,threads=make_uint3(tpb,1,1);
  CUTRY(cudaGetDeviceProperties(&prop,0));
  DBG("MAX GRID: %7d %7d %7d"ENDL,prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
  // Pack our 1d indexes into cuda's 3d indexes
  TRY(grid.x=nextdim(blocks,prop.maxGridSize[0],&r));
  blocks/=grid.x;
  blocks+=r;
  TRY(grid.y=nextdim(blocks,prop.maxGridSize[1],&r));
  blocks/=grid.y;
  blocks+=r;
  TRY(grid.z=blocks);
  DBG("    GRID: %7d %7d %7d"ENDL,grid.x,grid.y,grid.z);
  /// @cond DEFINES
  #define CASE2(TSRC,TDST)  DBG("blocks:%u threads/block:%u\n",b,tpb);\
                            affine_kernel<TSRC,TDST><<<grid,threads>>>(dst,src,transform,*param);\
                            break  
  #define CASE(T) TYPECASE2(ndtype(dst_),T); break
  /// @endcond
  TYPECASE(ndtype(src_));
  #undef CASE
  #undef CASE2
  CUTRY(cudaGetLastError());
  return 1;
Error:
  return 0;
}