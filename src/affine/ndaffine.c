/**
 * \file
 * Continuous nd affine transform.
 */
#include "nd.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _MSC_VER
#define alloca _alloca
#endif

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

#ifndef restrict
#define restrict __restrict__
#endif

extern unsigned ndaffine_cuda(nd_t dst_, const nd_t src_, const double *transform, const nd_affine_params_t *param);
/// @endcond

// import kind capabilities
#include "../private/kind.c"

//
// === CPU based implementation ===
//

/**
 * Bounds checking for \a pos.
 * \returns 1 if in-bounds, 0 otherwise
 */
static unsigned inbounds(const size_t nd, const size_t *restrict const shape, const double *restrict const pos)
{ size_t i;
  unsigned out=1;
  for(i=0;i<nd;++i) out*=(0<=pos[i] && pos[i]<shape[i]);
#if 0
  if(!out) printf("OUT OF BOUNDS\n");
#endif
  return out;
}

/**
 * Affine projection <tt>z=T.x</tt>
 * \verbatim
 * [z 1] = [m b] * [x]
 *         [0 1]   [1]
 * \endverbatim
 * 
 * \param[in,out] z   Output vector.
 * \param[in]     nz  Number of elements in \a z.
 * \param[in]     T   A (\a nz+1)-by-(\a nx+1) row-major affine projection matrix.
 * \param[in]     x   Input vector.
 * \param[in]     nx  Number of elements in \a x.
 */
static void proj(      double *restrict z, const size_t nz,
                 const double *restrict const T,
                 const size_t *restrict x, const size_t nx)
{ const size_t NC=nx+1;
  size_t r,c;
  memset(z,0,sizeof(double)*nz);
  for(r=0;r<nz;++r)
  { for(c=0;c<nx;++c)  // rotation,scale,shear
      z[r]+=T[r*NC+c]*x[c];
    z[r]+=T[r*NC+nx];  // translation
  }
}

/**
 * Iterate through voxel positions.
 * Inner dimensions are iterated first.
 * \returns 0 when done iterating, 1 otherwise.
 */
static unsigned inc(const size_t nd,
                    const size_t *restrict const shape,
                    const size_t *restrict const strides,
                    u8 **ptr,
                    size_t *restrict pos)
{ size_t d=0;
  const size_t s=strides[0];
  while(d<nd && pos[d]==shape[d]-1) //carry
    pos[d++]=0;
  if(d>=nd) return 0;
  pos[d]++;
  (*ptr)+=s;
#if 0
  { size_t i;
    for(i=0;i<nd;++i)
      printf("%3u",(unsigned)pos[i]);
    printf("[%2u:%9u]\n",(unsigned)d,(unsigned)strides[d]);
  }  
#endif
  return 1;
}

// Import generics
#include "src/affine/generic/all.c"
#include "src/affine/generic/macros.h"

/**
 * Generic affine transform on the cpu.
 * \see ndaffine()
 */
static unsigned ndaffine_cpu(nd_t dst, const nd_t src, const double *transform, const nd_affine_params_t *param)
{ /// @cond DEFINES
  #define CASE2(T1,T2) return ndaffine_cpu_##T1##_##T2(dst,src,transform,param);
  #define CASE(T)      TYPECASE2(ndtype(dst),T); break
  /// @endcond
      TYPECASE(ndtype(src));
  #undef CASE
  #undef CASE2
  return 1;
Error:
  return 0;
}

//
// === INTERFACE ===
//

// struct nd_affine_params_t
// { double boundary_value;
// };

/**
 * \todo interpolation
 * \todo boundary conditions
 * \todo anti-alias filter
 * 
 * \param[in,out]   dst         Should have the same ndkind() as \a src.
 * \param[in]       src         Acceptable ndkind()'s are those that support
 *                              pointer arithmetic and, optionally, CUDA.
 * \param[in]       transform   Matrix representing the transform mapping
 *                              coordinates (voxels) in the \a dst space to the
 *                              \a src space.
 *                              A row-major matrix of dimension
 *                              <tt>[ndndim(src),ndndim(dst)]</tt>
 *
 * \returns dst on success, 0 otherwise.
 */
nd_t ndaffine(nd_t dst, const nd_t src, const double *transform, const nd_affine_params_t* params)
{ REQUIRE(src,PTR_ARITHMETIC);
  TRY(ndkind(dst)==ndkind(src));
  //TRY(ndAntialiasFilter(src,transform));
  switch(ndkind(src))
  { case nd_gpu_cuda: TRY(ndaffine_cuda(dst,src,transform,params)); break;
    default:          TRY(ndaffine_cpu (dst,src,transform,params)); break;
  }
  return dst;
Error:
  return 0;
}