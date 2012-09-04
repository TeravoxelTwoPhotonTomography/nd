/**
 * \file
 * nd affine transform on the GPU with CUDA.
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
#define WARPS_PER_BLOCK  9
#define BLOCKSIZE       (32*WARPS_PER_BLOCK) // threads per block

#define ENDL "\n"
#define LOG(...) ndLogError(dst_,__VA_ARGS__)
#define CUTRY(e) do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,cudaGetErrorString(ecode)); goto Error; }}while(0)
#define FAIL     LOG("%s(%d) %s()"ENDL "\tExecution should not have reached here."ENDL,__FILE__,__LINE__,__FUNCTION__); goto Error

#define shared extern "C"
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

inline __device__ float fpartf(float f) { return f-(long)f;}

inline __device__ u8 inbounds_(float x,size_t n)
{return floorf(x)>=0.0f && floorf(x)<n;}

inline __device__ uchar2 inbounds(u8 ndim,const size_t*restrict const shape, const float*restrict const r)
{ uchar2 b=make_uchar2(0,0);
  for(u8 i=0;i<ndim;++i)
  { b.x|=(inbounds_(r[i]     ,shape[i])<<i);
    b.y|=(inbounds_(r[i]+1.0f,shape[i])<<i);
  }
  return b;
}

/**
 * nD linear interpolation for maximum intensity composting.
 *
 * The boundary handling used here is designed for maximum intensity composting.
 * A constant (determined by \a param->boundary_value) is returned for 
 * out-of-bounds samples.
 * Samples stradling the border are handled as a special case.
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

/**
 * Yield a position vector from an index.
 * For r=(x,y,z...) in a box with dimensions (Nx,Ny,Nz,..)
 * idx = x+Nx(y+Ny*(z+Nz(...)))
 */
inline __device__ void idx2pos(u8 ndim, const size_t *restrict const shape, unsigned idx, unsigned *restrict r)
{ for(u8 i=0;i<ndim;++i)
  { r[i]=idx%shape[i];
    idx/=shape[i];
  }
}

/**
 * Transform input vector according to an affine projection matrix.
 * \verbatim
 *             T
 * [lhs 1] = [m b] * [rhs]
 *           [0 1]   [1  ]
 * \endverbatim
 * 
 * \param[in,out] lhs  Output vector (left-hand side).
 * \param[in]     nlhs Number of elements in \a lhs.
 * \param[in]     T    A (\a nlhs+1)-by-(\a nrhs+1) row-major affine projection matrix.
 * \param[in]     rhs  Input vector (right-hand side).
 * \param[in]     nrhs Number of elements in \a rhs.
 */
inline __device__ void proj(
           float *restrict       lhs,
              u8                 nlhs,
  const   double *restrict const T,
  const unsigned *restrict const rhs,
              u8                 nrhs
  )
{ for(unsigned r=0;r<nlhs;++r)
  { lhs[r]=0.0f;
    for(unsigned c=0;c<nrhs;++c)
      lhs[r]+=rhs[c]*T[(nrhs+1)*r+c];
    lhs[r]+=T[(nrhs+1)*r+nrhs];
  }
}

#define max(a,b) ((a)>(b))?(a):(b)

template<typename Tsrc,typename Tdst> 
__global__ void affine_kernel(arg_t dst, arg_t src, const double *transform, const nd_affine_params_t param)
{ 
  Tdst     obuf=0;
  Tdst     ibuf=0;  
  unsigned rdst[MAXDIMS];
  float    rsrc[MAXDIMS];
  unsigned idst=threadIdx.x+blockIdx.x*blockDim.x;
#if 0
  if(blockIdx.x==0 && threadIdx.x==2)
    printf("ksize src:%d dst:%d\n",(int)sizeof(*ibuf),(int)sizeof(*obuf));
#endif
  if(idst<dst.nelem)
  { idx2pos(dst.ndim,dst.shape,idst,rdst);
    proj(rsrc,src.ndim,transform,rdst,dst.ndim);
    ibuf=sample<Tsrc,Tdst>(src,rsrc,&param);
    obuf=((Tdst*)dst.data)[idst];
    __syncthreads();
//    ((Tdst*)dst.data)[idst]=ibuf;
    if(ibuf>obuf)
      ((Tdst*)dst.data)[idst]=ibuf;
    else
      ((Tdst*)dst.data)[idst]=obuf;
//    ((Tdst*)dst.data)[idst]=max(obuf,ibuf);
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



/**
 * Assume the ndkind() of \a src_ and \a dst_ have already been checked.
 */
shared unsigned ndaffine_cuda(nd_t dst_, const nd_t src_, const double *transform, const nd_affine_params_t *param)
{ arg_t dst=make_arg(dst_),
        src=make_arg(src_);
  /// @cond DEFINES
  #define CASE2(TSRC,TDST)  printf("size src:%d dst:%d\n",(int)sizeof(TSRC),(int)sizeof(TDST));affine_kernel<TSRC,TDST><<<1+(unsigned)dst.nelem/BLOCKSIZE,BLOCKSIZE,0,0>>>(dst,src,transform,*param); break  
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