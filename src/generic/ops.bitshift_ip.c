/** \file
    Implementation of generic, in-place bitshift over an array.

    Floating point types aren't normally ammenable to bitshift's.  Here, floats and
    doubles are cast to a wide enough unsigned integer and then bitshift'd.  The
    result is cast back to a float or double.

    This should not be built with the main project.  Instead include it at the
    top of the c file defining the public-facing interface for these
    algorithms.

    To get the definitions for integral types, \a TDST, must be defined that
    names the basic voxel type.

    It should look something like this:
    \code
    #define TDST uint8_t
    #include "generic/ops.bitshift_ip.c" // will undef TDST at the end
    #define TDST uint16_t
    #include "generic/ops.bitshift_ip.c"
    ...
    \endcode

    This will produce function definitions like:
    \code
    bitshift_ip_uint8_t(...)
    bitshift_ip_uint16_t(...)
    \endcode

    The implementations for floating point types are specialized.  To get those
    define \a TSPECIAL.  For example:
    \code
    #define TSPECIAL
    #include "generic/ops.bitshift_ip.c"
    \encode

    This will produce:
    \code
    bitshift_ip_f32(...)
    xpr_ip_f64(...)
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

/** In-place bitshift with a overflow mask.
 *  \code
 *     z = (z<<b)&m
 *  \endcode
 *  where \c b=param[0] and \c m=~(1<<param[1]).
 *
 *  \param[in]     N   The number of elements on which to operate.
 *  \param[out]    z   The output buffer.
 *  \param[in]     zst The number of bytes between each element in \a z.
 *  \param[in]   param The parameters of the bitshift as two successive ints.
 *  \param[in]  nbytes The number of bytes in \a param (ignored).
 */
static void NAME(bitshift_ip,TDST)(stride_t N,
                              void* restrict z,stride_t zst,
                              void* restrict param, size_t nbytes)
{ TDST * restrict _z = (TDST*)z;
  int *p = (int*)param;
  TDST mask=(((TDST)1)<<p[1])-1;
  const size_t _zst = zst/sizeof(TDST);
  { size_t i;
    int b;
    if(p[0]<0)
    { b=-p[0];
      for(i=0;i<N;++i) _z[i*_zst]=(_z[i*_zst]>>b)&mask;
    } else
    { b=p[0];
      for(i=0;i<N;++i) _z[i*_zst]=(_z[i*_zst]<<b)&mask;
    }
  }
}
#undef TDST
#undef NAME
#undef _NAME
