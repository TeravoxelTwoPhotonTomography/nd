/** \file
    Implementation of generic, in-place scalar fmad (floating multiply-and-add) 
    over an array.

    This should not be built with the main project.  Instead include it at the
    top of the c file defining the public-facing interface for these
    algorithms.

    To get the definitions for integral types, \a TDST, must be defined that
    names the basic voxel type.

    It should look something like this:
    \code
    #define TDST uint8_t
    #include "generic/ops.fmad_scalar_ip.c" // will undef TDST at the end
    #define TDST uint16_t
    #include "generic/ops.fmad_scalar_ip.c"
    ...
    \endcode

    This will produce function definitions like:
    \code
    fmad_scalar_ip_uint8_t(...)
    fmad_scalar_ip_uint16_t(...)
    \endcode

    \author Nathan Clack
    \date   2012
*/
#include <string.h>

#ifndef TDST
#error "TDST must be defined"
#endif

#ifndef restrict
#define restrict __restrict
#endif

#ifndef NAME
#define NAME(ROOT,a)  _NAME(ROOT,a)  ///< Macro for c-style "templated" functions
#define _NAME(ROOT,a) ROOT##_##a     ///< Macro for c-style "templated" functions
#endif

#undef max
#undef min
#define max(a,b) (((a)<(b))?(b):(a))
#define min(a,b) (((a)<(b))?(a):(b))

/**
 * Saturation
 *
 * This defines what happens when the source value overflows the destination
 * type.
 */
static TDST NAME(saturate,TDST)(f32 v)
{ return min(max(v,NAME(min,TDST)),NAME(max,TDST));
}

/** In-place xor with a constant.
 *  \code
 *     z = m*z+b
 *  \endcode
 *  where \c c=param[0].
 *
 *  \param[in]     N   The number of elements on which to operate.
 *  \param[out]    z   The output buffer.
 *  \param[in]     zst The number of bytes between each element in \a z.
 *  \param[in]   param A two-element f32 array with <tt>[m,b]</tt>.
 *  \param[in]  nbytes The number of bytes in \a param (ignored).
 */
static void NAME(fmad_scalar_ip,TDST)(stride_t N,
                              void* restrict z,stride_t zst,
                              void* restrict param, size_t nbytes)
{ TDST * restrict _z = (TDST*)z;
  f32 *p = (f32*)param;
  const f32 m=p[0],b=p[1];
  const size_t _zst = zst/sizeof(TDST);
  { size_t i;
    for(i=0;i<N;++i)
      _z[i*_zst]=NAME(saturate,TDST)(m*_z[i*_zst]+b);
  }
}
#undef TDST
#undef NAME
#undef _NAME


