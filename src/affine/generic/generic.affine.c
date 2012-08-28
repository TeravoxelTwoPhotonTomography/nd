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

#ifndef TSRC
#error "TSRC must be defined"
#endif

#ifndef restrict
#define restrict __restrict
#endif

#ifndef NAME
#define NAME(ROOT,a,b)  _NAME(ROOT,a,b)      ///< Macro for c-style "templated" functions
#define _NAME(ROOT,a,b) ROOT##_##a##_##b     ///< Macro for c-style "templated" functions
#endif

/**
 * Nearest neighbor sampling.
 */
static TDST NAME(sample,TSRC,TDST)(const size_t ndims, 
                                   const size_t *restrict const shape,
                                   const size_t *restrict const strides,
                                   const void   *restrict const data,
                                   const double *restrict const pos,
                                   const nd_affine_params_t *restrict const param)
{ size_t i;  
  const u8 *restrict d=(u8*)data;
  if(!inbounds(ndims,shape,pos)) return (TDST)param->boundary_value;
  for(i=0;i<ndims;++i)
    d+=strides[i]*(size_t)pos[i];
  return *(TDST*)d;
}

// static unsigned inc(const size_t nd,
//                     const size_t *restrict const shape,
//                     const size_t *restrict const strides,
//                     u8 **ptr,
//                     size_t *restrict pos)

/**
 * Affine transform.  Type specific implementation.
 * \see ndaffine()
 */
static unsigned NAME(ndaffine_cpu,TSRC,TDST)(
  nd_t dst, 
  const nd_t src, 
  const double *restrict transform,
  const nd_affine_params_t *restrict param)
{ const size_t ndd=ndndim(dst),
           *dshape=ndshape(dst),
         *dstrides=ndstrides(dst),
               nds=ndndim(src),
           *sshape=ndshape(src),
         *sstrides=ndstrides(src);
  const void*restrict sdata=nddata(src);
  TDST *restrict ddata=(TDST*)nddata(dst);
  size_t *dstpos;
  double *srcpos;
  TRY(dstpos=alloca(sizeof(*dstpos)*ndd));
  TRY(srcpos=alloca(sizeof(*srcpos)*nds));
  memset(dstpos,0,sizeof(*dstpos)*ndd);
  do
  { TDST v;
    proj(srcpos,nds,transform,dstpos,ndd);
    v=NAME(sample,TSRC,TDST)(nds,sshape,sstrides,sdata,srcpos,param);
    *ddata=MAX(*ddata,v);
  } while(inc(ndd,dshape,dstrides,(u8**)&ddata,dstpos));
  return 1;
Error:
  return 0;
}

#undef TDST