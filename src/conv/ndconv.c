/**
 * \file
 * nD direct (not fft-based) convolutions.
 */

#include "nd.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/// @cond PRIVATE
#define ENDL      "\n"
#define LOG(...)  ndLogError(dst,__VA_ARGS__)
#define TRY(e)    do{if(!(e)) { LOG("%s(%d): %s"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error;}} while(0)
#define TRYMSG(e,msg) do{if(!(e)) { LOG("%s(%d): %s"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,msg); goto Error;}} while(0)
#define FAIL      do{ LOG("%s(%d):"ENDL "\tExecution should not reach here."ENDL,__FILE__,__LINE__); goto Error;} while(0)

#define MAX(a,b)  (((a)>(b))?(a):(b))

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef  int8_t  i8;
typedef  int16_t i16;
typedef  int32_t i32;
typedef  int64_t i64;
typedef float    f32;
typedef double   f64;

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
#define max_f64  FLT_MAX

#undef max
#undef min
#define max(a,b) (((a)<(b))?(b):(a))
#define min(a,b) (((a)<(b))?(a):(b))

#ifndef restrict
#define restrict __restrict__
#endif

extern unsigned ndconv1_cuda(nd_t dst,nd_t src,const nd_t filter,const unsigned idim,const nd_conv_params_t *param);
/// \todo extern unsigned ndconv1_ip_cuda(nd_t dst, const nd_t filter, const unsigned idim, const nd_conv_params_t *param);

/// @endcond

// import kind capabilities
#include "../private/kind.c"

// returns a pointer to the boundary value to use
typedef void *(*boundary_t)(int64_t i,void* data,size_t stride,size_t n,const nd_conv_params_t* params);


static void *replicate(int64_t i,void* data,size_t stride,size_t n,const nd_conv_params_t* params)
{ if(i<0)           return data;
  if(i>=(int64_t)n) return ((u8*)data)+stride*(n-1);
  return NULL;  // fail fast if called for in-bounds value.
}

static void *symmetric(int64_t i,void* data,size_t stride,size_t n,const nd_conv_params_t* params)
{ if(i<0)           return ((u8*)data)+stride*(-i);
  if(i>=(int64_t)n) return ((u8*)data)+stride*(2*n-1-i); // (n-1)-(i-n)
  return NULL;}

static void *circular(int64_t i,void* data,size_t stride,size_t n,const nd_conv_params_t* params)
{ int64_t m=i/n;
  m-=(m<0);
  return ((u8*)data)+stride*(i-m*n); // return data[mod(i,n)]
}

static void *zero(int64_t i,void* data,size_t stride,size_t n,const nd_conv_params_t* params)
{ static const u64 z=0; // need to have enough zero bits to cover each type
  return (void*)&z;
}

boundary_t select_boundary_condition(const nd_conv_params_t* params)
{ switch(params->boundary_condition)
  { case nd_boundary_replicate: return replicate;
    case nd_boundary_symmetric: return symmetric;
    case nd_boundary_circular:  return circular;
    case nd_boundary_zero:      return zero;
    default:                    return NULL;
  }
}

/**
 * Iterate through voxel positions.
 * Inner dimensions are iterated first.
 * \returns 0 when done iterating, 1 otherwise.
 */
static unsigned inc(const size_t ndims,
                    const size_t idim, ///< exclude this dimension from iteration
                    const size_t *restrict const shape,
                    const size_t *restrict const strides,
                    u8 **ptr,
                    size_t *restrict pos)
{ size_t d=(idim==0); // skip idim if idim=0
  size_t s=strides[idim==0];
  while(d<ndims && pos[d]==shape[d]-1) //carry
  { pos[d++]=0;
    s+=(d==idim)?(strides[idim+1]-strides[idim]):0;
    d+=(d==idim);     //skip idim
  }
  if(d>=ndims) return 0;  
  pos[d]++;
  (*ptr)+=s;          // move ptr to pos
  return 1;
}


// Import generics
#include "src/conv/generic/all.c"
#include "src/conv/generic/macros.h"

/**
 * Generic out-of-place 1d convolution transform on the cpu.
 * Implementation first copies data to dst, and then does the in-place
 * 1d convolution.
 * 
 * \see ndconv1_ip()
 */
static unsigned ndconv1_cpu(nd_t dst, nd_t src, const nd_t filter, const unsigned idim, const nd_conv_params_t *param)
{ TRY(ndcopy(dst,src,0,0));
  TRY(ndconv1_ip(dst,filter,idim,param));
  return 1;
Error:
  return 0;
}

/**
 * Generic in-place 1d convolution transform on the cpu.
 * \see ndconv1_ip()
 */
static unsigned ndconv1_ip_cpu(nd_t dst, const nd_t filter, const unsigned idim, const nd_conv_params_t *param)
{ /// @cond DEFINES
  #define CASE2(T1,T2) return ndconv1_ip_cpu_##T1##_##T2(dst,filter,idim,param);
  #define CASE(T)      TYPECASE2(ndtype(dst),T); break
  /// @endcond
      TYPECASE(ndtype(filter));
  #undef CASE
  #undef CASE2
  return 1;
Error:
  return 0;
}

/**
 * 1D Out-of-place convolution against an nD array for performing seperable 
 * convolutions.
 */
nd_t ndconv1(nd_t dst, nd_t src, const nd_t filter, const unsigned idim,const nd_conv_params_t* params)
{ REQUIRE(dst,PTR_ARITHMETIC);
  REQUIRE(src,PTR_ARITHMETIC);
  TRY(ndndim(filter)==1);
  TRY(idim<ndndim(dst));
  switch(ndkind(dst))
  { case nd_gpu_cuda:
      REQUIRE(src,CAN_CUDA);
      REQUIRE(filter,CAN_MEMCPY); // must use a host pointer at the moment (no reason this can't be changed)
      TRY(ndconv1_cuda(dst,src,filter,idim,params)); 
      break;
    case nd_heap:
    case nd_static:
      TRY(ndconv1_cpu (dst,src,filter,idim,params)); break;
    default: FAIL;
  }
  return dst;
Error:
  return 0;
}

/**
 * 1D In-place convolution against an nD array for performing seperable 
 * convolutions.
 */
nd_t ndconv1_ip(nd_t dst, const nd_t filter, const unsigned idim,const nd_conv_params_t* params)
{ REQUIRE(dst,PTR_ARITHMETIC);
  TRY(ndndim(filter)==1);
  TRY(idim<ndndim(dst));
  switch(ndkind(dst))
  { case nd_gpu_cuda: 
      REQUIRE(filter,CAN_MEMCPY); // must use a host pointer at the moment (no reason this can't be changed)
      FAIL; //IN-PLACE NOT IMPLEMENTED: TRY(ndconv1_ip_cuda(dst,filter,idim,params)); 
      break;
    case nd_heap:
    case nd_static:   TRY(ndconv1_ip_cpu (dst,filter,idim,params)); break;
    default: FAIL;
  }
  return dst;
Error:
  return 0;
}