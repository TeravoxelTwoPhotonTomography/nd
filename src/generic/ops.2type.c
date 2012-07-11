/** \file
    Basic type generic operations on nD arrays that involve two types.

    \author Nathan Clack
    \date   2012

    This should not be built with the main project.
    Instead include it at the top of the c file defining the
    public-facing interface for these algorithms.

    Before including this file two macros, \a TSRC \a TDST must be defined that
    names the basic source and destination types.

    It should look something like this:
    \code
    #define TSRC uint8_t
    #define TDST uint8_t
    #include "generic/ops.2type.c" // will undef TDST at the end
    #define TDST uint16_t
    #include "generic/ops.2type.c"
    ...
    #undef TSRC
    \endcode

    This will produce function definitions like:
    \code
    copy_uint8_t_uint8_t(...)
    copy_uint8_t_uint16_t(...)
    \endcode
*/
#include <string.h>

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
#define NAME(ROOT,a,b)  _NAME(ROOT,a,b)
#define _NAME(ROOT,a,b) ROOT##_##a##_##b
#endif

static void NAME(copy,TSRC,TDST)(stride_t N,
                                 void* restrict z,stride_t zst,
                                 const void* restrict x,stride_t xst,
                                 void *param, size_t nbytes)
{ TSRC * restrict _x = (TSRC*)x;
  TDST * restrict _z = (TDST*)z;
  const size_t _xst = xst/sizeof(TSRC),
               _zst = zst/sizeof(TDST);
  if(sizeof(TSRC)==sizeof(TDST) && zst==xst)
    memcpy(z,x,xst*N);
  else
  { size_t i;
    for(i=0;i<N;++i)
      _z[i*_zst]=(TDST)_x[i*_xst];
  }
}

/** Projection operation.
 *  \code
 *    z = a*x+b*y
 *  \endcode
 *  where \c a=param[0] and \c b=param[1].
 *
 *  Operations are performed in \a TSRC space and then cast to \a TDST.
 *
 *  \tparam TSRC The type of the source array.
 *  \tparam TDST The type of the destination array.
 *  \param[in]     N   The number of elements on which to operate.
 *  \param[out]    z   The output buffer.
 *  \param[in]     zst The number of bytes between each element in \a z.
 *  \param[in]     x   The first input buffer.
 *  \param[in]     xst The number of bytes between each element in \a x.
 *  \param[in]     y   The first input buffer.
 *  \param[in]     yst The number of bytes between each element in \a y.
 *  \param[in]   param The coefficients for projection with type \a TSRC.
 *  \param[in]  nbytes The number of bytes in \a param (ignored).
 */
static void NAME(project,TSRC,TDST)(stride_t N,
                                 void* restrict z,stride_t zst,
                                 const void* restrict x,stride_t xst,
                                 const void* restrict y,stride_t yst,
                                 void *param, size_t nbytes)
{ TSRC * restrict _x = (TSRC*)x;
  TSRC * restrict _y = (TSRC*)y;
  TDST * restrict _z = (TDST*)z;
  const size_t _xst = xst/sizeof(TSRC),
               _yst = yst/sizeof(TSRC),
               _zst = zst/sizeof(TDST);
#if 1
  if(nbytes!=2*sizeof(TSRC))
    goto Error;
#endif
  { TSRC *p=(TSRC*)param,
          a=p[0],
          b=p[1];
    size_t i;
    for(i=0;i<N;++i)
      _z[i*_zst]=(TDST)(a*_x[i*_xst]+b*_y[i*_yst]);
  }
  return;
Error:
  exit(1);
}

/** Floating multiply and add.
 *  Just like \a project but with floating-point coefficients.
 *  \code
 *    z = a*x+b*y
 *  \endcode
 *  where \c a=param[0] and \c b=param[1].
 *
 *  Operations are performed in floating-point and then cast to \a TDST.
 *
 *  \tparam TSRC The type of the source array.
 *  \tparam TDST The type of the destination array.
 *  \param[in]     N   The number of elements on which to operate.
 *  \param[out]    z   The output buffer.
 *  \param[in]     zst The number of bytes between each element in \a z.
 *  \param[in]     x   The first input buffer.
 *  \param[in]     xst The number of bytes between each element in \a x.
 *  \param[in]     y   The first input buffer.
 *  \param[in]     yst The number of bytes between each element in \a y.
 *  \param[in]   param The coefficients for projection with type \a float.
 *  \param[in]  nbytes The number of bytes in \a param (ignored).
 */
static void NAME(fmad,TSRC,TDST)(stride_t N,
                                 void* restrict z,stride_t zst,
                                 const void* restrict x,stride_t xst,
                                 const void* restrict y,stride_t yst,
                                 void *param, size_t nbytes)
{ TSRC * restrict _x = (TSRC*)x;
  TSRC * restrict _y = (TSRC*)y;
  TDST * restrict _z = (TDST*)z;
  const size_t _xst = xst/sizeof(TSRC),
               _yst = yst/sizeof(TSRC),
               _zst = zst/sizeof(TDST);
#if 1
  if(nbytes!=2*sizeof(float))
    goto Error;
#endif
  { float *p=(float*)param,
          a=p[0],
          b=p[1];
    size_t i;
    for(i=0;i<N;++i)
      _z[i*_zst]=(TDST)(a*_x[i*_xst]+b*_y[i*_yst]);
  }
  return;
Error:
  exit(1);
}

#undef TDST
#undef restrict
#undef NAME
#undef _NAME
