/** \file
    Implementation of generic, in-place xor over an array.

    Floating point types aren't normally ammenable to xor's.  Here, floats and
    doubles are cast to a wide enough unsigned integer and then xor'd.  The
    result is cast back to a float or double.

    This should not be built with the main project.  Instead include it at the
    top of the c file defining the public-facing interface for these
    algorithms.

    To get the definitions for integral types, \a TDST, must be defined that
    names the basic voxel type.

    It should look something like this:
    \code
    #define TDST uint8_t
    #include "generic/ops.xor_ip.c" // will undef TDST at the end
    #define TDST uint16_t
    #include "generic/ops.xor_ip.c"
    ...
    \endcode

    This will produce function definitions like:
    \code
    xor_ip_uint8_t(...)
    xor_ip_uint16_t(...)
    \endcode

    The implementations for floating point types are specialized.  To get those
    define \a TSPECIAL.  For example:
    \code
    #define TSPECIAL
    #include "generic/ops.xor_ip.c"
    \encode

    This will produce:
    \code
    xor_ip_f32(...)
    xpr_ip_f64(...)
    \endcode

    \author Nathan Clack
    \date   2012
*/
#include <string.h>


#ifdef TSPECIAL
static void xor_ip_f32(stride_t N,
                       void* restrict z,stride_t zst,
                       void* restrict param, size_t nbytes)
{ f32 * restrict _z = (f32*)z;
  u64 p = *(u64*)param;
  const size_t _zst = zst/sizeof(f32);
  { size_t i;
    for(i=0;i<N;++i)
      _z[i*_zst]=(f32) (((u64)_z[i*zst])^p);
  }
}
static void xor_ip_f64(stride_t N,
                       void* restrict z,stride_t zst,
                       void* restrict param, size_t nbytes)
{ f64 * restrict _z = (f64*)z;
  u64 p = *(u64*)param;
  const size_t _zst = zst/sizeof(f64);
  { size_t i;
    for(i=0;i<N;++i)
      _z[i*_zst]=(f64) (((u64)_z[i*zst])^p);
  }
}
#undef TSPECIAL
#else // defined TSPECIAL



#ifndef TDST
#error "TDST must be defined"
#endif

#ifndef restrict
#define restrict __restrict
#endif

#ifndef NAME
#define NAME(ROOT,a)  _NAME(ROOT,a)
#define _NAME(ROOT,a) ROOT##_##a
#endif

/** In-place xor with a constant.
 *  \code
 *     z ^= c
 *  \endcode
 *  where \c c=param[0].
 *
 *  \param[in]     N   The number of elements on which to operate.
 *  \param[out]    z   The output buffer.
 *  \param[in]     zst The number of bytes between each element in \a z.
 *  \param[in]   param The bits against which to xor as an unsigned 64-bit integer.
 *  \param[in]  nbytes The number of bytes in \a param (ignored).
 */
static void NAME(xor_ip,TDST)(stride_t N,
                              void* restrict z,stride_t zst,
                              void* restrict param, size_t nbytes)
{ TDST * restrict _z = (TDST*)z;
  u64 p = *(u64*)param;
  const size_t _zst = zst/sizeof(TDST);
  { size_t i;
    for(i=0;i<N;++i)
      _z[i*_zst]^=p;
  }
}
#undef TDST
#undef NAME
#undef _NAME
#endif


