/** \file
 *  Generic nd affine transfom on the cpu.
 *
 *  This should not be built with the main project.  Instead include it at the
 *  top of the c file defining the public-facing interface for these
 *  algorithms.
 *
 *  Before including this file two macros, \a TSRC \a TDST must be defined that
 *  names the basic source and destination types.
 *
 *  It should look something like this:
 *  \code
 *  #define TSRC uint8_t
 *  #define TDST uint8_t
 *  #include "generic/affine.c" // will undef TDST at the end
 *  #define TDST uint16_t
 *  #include "generic/affine.c"
 *  ...
 *  #undef TSRC
 *  \endcode
 *
 *  This will produce function definitions like:
 *  \code
 *  affine_cpu_uint8_t_uint8_t(...)
 *  affine_cpu_uint8_t_uint16_t(...)
 *  \endcode
 *
 *  \see ndaffine.c
 *  The generic function defined here are used in ndaffine_cpu().
 * 
 *  \author Nathan Clack
 *  \date   2012
 */

#ifndef TDST
#error "TDST must be defined"
#endif

#ifndef restrict
#define restrict __restrict
#endif

#ifndef NAME
#define NAME(ROOT,a,b)  _NAME(ROOT,a,b)      ///< Macro for c-style "templated" functions
#define _NAME(ROOT,a,b) ROOT##_##a##_##b     ///< Macro for c-style "templated" functions
#define NAME1(ROOT,a)   _NAME1(ROOT,a)       ///< Macro for c-style "templated" functions
#define _NAME1(ROOT,a)  ROOT##_##a           ///< Macro for c-style "templated" functions
#endif

/**
 * Saturation
 *
 * This defines what happens when the source value overflows the destination
 * type.
 */
static TDST NAME(saturate,TSRC,TDST)(TSRC v)
{ return min(max((double)v,NAME1(min,TDST)),NAME1(max,TDST));
}


  
  // static unsigned inc(const size_t ndims,
  //                     const size_t idim, ///< exclude this dimension from iteration
  //                     const size_t *restrict const shape,
  //                     const size_t *restrict const strides,
  //                     u8 **ptr,
  //                     size_t *restrict pos) 

/**
 * Affine transform.  Type specific implementation.
 * Assumptions about the input should be checked by the caller.
 *
 * 1. filter should be 1d
 * 2. filter should have odd length, origin is at center (floor(N/2))
 * 
 * 
 * Addition/multiplication is done in the filter data type before
 * being cast to the destination type.
 *
 * Uses ndshape(dst)[idim] memory (from heap) as temporary space to do the 
 * computation in-place.  Could use less (just need the filter window), but
 * doing the simple thing first.
 * 
 * \see ndaffine()
 */
static unsigned NAME(ndconv1_ip_cpu,TSRC,TDST)(
  nd_t dst, 
  const nd_t filter, 
  const unsigned idim,
  const nd_conv_params_t *restrict param)
{ const int64_t CF=ndnelem(filter),
                half=CF/2,
                CD=ndshape(dst)[idim],
                SD=ndstrides(dst)[idim];
  boundary_t boundary=select_boundary_condition(param);
  TDST *restrict t=0,
       *restrict d=nddata(dst);
  TSRC *restrict f=((TSRC*)nddata(filter))+half;

  size_t *dstpos;
  TRY(dstpos=malloc(sizeof(*dstpos)*ndndim(dst)));
  memset(dstpos,0,sizeof(*dstpos)*ndndim(dst));

  // alloc temp row, t
  TRY(t=(TDST*)malloc(CD*sizeof(TDST)));
  do
  { int64_t i;
    for(i=0;i<CD;++i)
    { TSRC v=0;
      int64_t j;
      // iterate half to -half (inclusive) going backwards
      // This will access dst in a forward sweep
      for(j=half;(j>i)&&(j>=-half);--j)
        v+=f[j]*(*(TDST*)boundary(i-j,d,SD,CD,param));
      for(;(j>i-CD)&&(j>=-half);--j)
        v+=f[j]*d[i-j];
      for(;j>=-half;--j)
        v+=f[j]*(*(TDST*)boundary(i-j,d,SD,CD,param));
      t[i]=NAME(saturate,TSRC,TDST)(v);
    }
    // copy t to dst
    memcpy(d,t,CD*sizeof(TDST));
  } while(inc(ndndim(dst),idim,ndshape(dst),ndstrides(dst),(u8**)&d,dstpos));
  free(t);
  return 1;
Error:
  if(t) free(t);
  return 0;
}

#undef TDST