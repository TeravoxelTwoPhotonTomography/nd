/** \file
    Basic nd-array algorithms

    \author Nathan Clack
    \date   June 2012

    \todo convert
    \todo refactor setting ndim and shape to it's own function
*/
#include <stdint.h>
#if defined(__APPLE__) || defined(__MACH__)
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif
#include <stdio.h>
#include "nd.h"
#include "ops.h"

#define restrict   __restrict
#define countof(e) (sizeof(e)/sizeof(*(e)))

#ifdef _MSC_VER
#define alloca _alloca
#endif

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

typedef size_t  stride_t;

typedef void (inplace_vec_op_t)(stride_t N, void *z, stride_t zst, void *param, size_t nbytes);
typedef void (unary_vec_op_t)(stride_t N,void* z,stride_t zst,const void* x,stride_t xst,void *param, size_t nbytes);                             ///< 1D f:z=f(x)
typedef void (binary_vec_op_t)(stride_t N,void* z,stride_t zst,const void* x,stride_t xst,const void* y,stride_t yst,void *param, size_t nbytes); ///< 1D f:z=f(x,y)

///// Import generics
#include "generic/all.c"
#include "generic/macros.h"

/////
///// Error handling
/////
#define ENDL     "\n"
#define LOG(...) fprintf(stderr,__VA_ARGS__)
#define TRY(e)   do{if(!(e)) {LOG("%s(%d): %s"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error; }}while(0)
#define FAIL     do{          LOG("%s(%d): %s"ENDL "\tExecution should not reach here."ENDL,__FILE__,__LINE__,__FUNCTION__); goto Error; }while(0)

/////
///// Basic Element-wise operations
/////

/** \returns the sign bit to flip if there's a signed/unsigned conversion,
 *  otherwise 0.
 */
static int get_sign_change_bit(nd_type_id_t dst, nd_type_id_t src)
{ static const int isSigned[]   = {0,0,0,0,1,1,1,1,1,1},
                   isFloating[] = {0,0,0,0,0,0,0,0,1,1},
                   bit[] = {7,15,31,63,7,15,31,63,0,0};
  return (isFloating[src]||isFloating[dst])?0
        :(  isSigned[src]==  isSigned[dst])?0
        :bit[src];
}

/**
   Find max index such that lower dimension indexes may be collapsed(vectorized)
   \verbatim
        ostrides[j]==istrides[j] and oshape[j]==ishape[j]==shape[j] for j<i
      where oshape and ishape are the shapes derived from the input
        strides.
  \endverbatim
*/
static stride_t find_vectorizable_dim(stride_t ndim, const stride_t *shape, int narrays, const stride_t **strides)
{ stride_t m;
  int i;
  stride_t *shapes;
  TRY(shapes=(stride_t*)alloca(sizeof(stride_t)*ndim*narrays));
  for(i=0;i<narrays;++i)
  { stride_t *s =shapes +i*ndim;
    const stride_t *st=strides[i];
    for(m=0;m<ndim;++m)            // compute native shapes from strides
      s[m]=st[m+1]/st[m];
  }

  for(m=0;m<ndim;++m)
  { for(i=0;i<narrays;++i)
    { stride_t *s =shapes +i*ndim;
      const stride_t *st=strides[i];
      if(   s[m]!=shape[m]
        || st[m]!=strides[0][m] )  // sufficient to check against first, since all must be same
        return m;
    }
  }
  return m;
Error:
  return -1;
}

///// IN-PLACE
static void inplace_op_recurse(
    int64_t m,
    int64_t idim,
    const stride_t *shape,
    void* z, const stride_t *zst,
    void* param, size_t nbytes,
    inplace_vec_op_t *f)
{
  if(idim<m)
  { f(zst[m]/zst[0],z,zst[0],param,nbytes);
    return;
  }
  if(idim<1)
  { f(shape[0],z,zst[0],param,nbytes);
    return;
  }
  { stride_t i;
    const stride_t oz = zst[idim];
    for(i=0;i<shape[idim];++i)
      inplace_op_recurse(m,idim-1,shape,(u8*)z+oz*i,zst,param,nbytes,f);
  }
}

/** Apply an in-place operation over an \a ndim-dimensional subvolume of \a z.
 *
 *  The extent of the subvolume is defined by \a shape.
 *  The implementation uses the same recursive pattern as unary_op() and binary_op().
 */
static int inplace_op(
    stride_t ndim,
    stride_t *shape,
    void* z,const stride_t *zst,
    void* param, size_t nbytes,
    inplace_vec_op_t *f)
{
  int64_t i,m;
  const stride_t *strides[] = {zst};
  TRY( (m=find_vectorizable_dim(ndim,shape,countof(strides),strides))>=0 );
  inplace_op_recurse(m,ndim-1,shape,z,zst,param,nbytes,f);
  return 1;
Error:
  return 0; // stack overflowed
}

///// UNARY
static void unary_op_recurse(
    int64_t m,
    int64_t idim,
    const stride_t *shape,
          void* z, const stride_t *zst,
    const void* x, const stride_t *xst,
    void* param, size_t nbytes,
    unary_vec_op_t *f)
{
  if(idim<m)
  { f(zst[m]/zst[0],z,zst[0],x,xst[0],param,nbytes);
    return;
  }
  if(idim<1)
  { f(shape[0],z,zst[0],x,xst[0],param,nbytes);
    return;
  }
  { stride_t i;
    const stride_t ox = xst[idim],
                   oz = zst[idim];
    for(i=0;i<shape[idim];++i)
      unary_op_recurse(m,idim-1,shape,(u8*)z+oz*i,zst,(u8*)x+ox*i,xst,param,nbytes,f);
  }
}

static int unary_op(
    stride_t ndim,
    stride_t *shape,
          void* z,const stride_t *zst,
    const void* x,const stride_t *xst,
    void* param, size_t nbytes,
    unary_vec_op_t *f)
{
  int64_t m;
  const stride_t *strides[] = {zst,xst};
  TRY( (m=find_vectorizable_dim(ndim,shape,countof(strides),strides))>=0 );
  unary_op_recurse(m,ndim-1,shape,z,zst,x,xst,param,nbytes,f);
  return 1;
Error:
  return 0; // stack overflowed
}

///// BINARY
static void binary_op_recurse(
    int64_t m,
    int64_t idim,
    const stride_t *shape,
          void* z,const stride_t *zst,
    const void* x,const stride_t *xst,
    const void* y,const stride_t *yst,
    void* param, size_t nbytes,
    binary_vec_op_t *f)
{
  if(idim<m)
  { f(zst[m]/zst[0],z,zst[0],x,xst[0],y,yst[0],param,nbytes);
    return;
  }
  if(idim<1)
  { f(shape[0],z,zst[0],x,xst[0],y,yst[0],param,nbytes);
    return;
  }
  { stride_t i;
    const stride_t ox = xst[idim],
                   oy = yst[idim],
                   oz = zst[idim];
    for(i=0;i<shape[idim];++i)
      binary_op_recurse(m,idim-1,shape,(u8*)z+oz*i,zst,(u8*)x+ox*i,xst,(u8*)y+oy*i,yst,param,nbytes,f);
  }
}

static int binary_op(
    stride_t ndim,
    stride_t *shape,
          void* z,const stride_t *zst,
    const void* x,const stride_t *xst,
    const void* y,const stride_t *yst,
    void* param, size_t nbytes,
    binary_vec_op_t *f)
{
  int64_t m;
  const stride_t *strides[] = {zst,xst,yst};
  TRY( (m=find_vectorizable_dim(ndim,shape,countof(strides),strides))>=0 );
  binary_op_recurse(m,ndim-1,shape,z,zst,x,xst,y,yst,param,nbytes,f);
  return 1;
Error:
  return 0; // stack overflowed
}

/////
///// INTERFACE OPS
/////

static size_t min_sz_t(size_t a, size_t b)
{ return (a<b)?a:b; }

#undef LOG
#define LOG(...) do{ ndLogError(dst,__VA_ARGS__); ndLogError(src,__VA_ARGS__); } while(0)

nd_t ndcopy(nd_t dst, const nd_t src, size_t ndim, size_t *shape)
{
  TRY(ndkind(src)==nd_cpu); // this implementation won't work for gpu based arrays
  // if shape or ndims is unspecified use smallest
  if(!ndim)
    ndim=min_sz_t(ndndim(dst),ndndim(src));
  if(!shape)
  { size_t i;
    TRY(shape=(size_t*)alloca(sizeof(size_t)*ndim));
    for(i=0;i<ndim;++i)
      shape[i]=min_sz_t(ndshape(dst)[i],ndshape(src)[i]);
  }
  #define CASE2(T1,T2) TRY(unary_op(ndim,shape, \
                              nddata(dst),ndstrides(dst), \
                              nddata(src),ndstrides(src), \
                              0,0, \
                              copy_##T1##_##T2)); break
  #define CASE(T)      TYPECASE2(ndtype(dst),T); break
      TYPECASE(ndtype(src));
  #undef CASE
  #undef CASE2
  // convert signed vals to unsigned vals in case of type change
  { int b;
    if(b=get_sign_change_bit(ndtype(dst),ndtype(src)))
      ndxor_ip(dst,1ULL<<b,ndim,shape);
  }
  return dst;
Error:
  return NULL;
}

#undef LOG
#define LOG(...) do{ ndLogError(z,__VA_ARGS__); ndLogError(x,__VA_ARGS__); ndLogError(y,__VA_ARGS__);} while(0)

nd_t ndadd(nd_t z, const nd_t x, const nd_t y, size_t ndim, size_t *shape)
{ nd_t args[] = {z,x,y};
  size_t i, bytesof_param;
  u8 param[8*2];
  TRY(ndkind(x)==ndkind(y));      // Require x and y have the same type, z type may vary
  for(i=0;i<countof(args);++i)
    TRY(ndkind(args[i])==nd_cpu);       // this implementation won't work for gpu based arrays
  // set shape and dim if necessary
  if(!ndim)
  { for(i=1,ndim=ndndim(args[0]);i<countof(args);++i)
      ndim=min_sz_t(ndim,ndndim(args[i]));
  }
  if(!shape)
  { TRY(shape=(size_t*)alloca(sizeof(size_t)*ndim));
    memcpy(shape,ndshape(args[0]),sizeof(size_t)*ndim);
    for(i=1;i<countof(args);++i)
    { size_t j;
      for(j=0;j<ndim;++j)
        shape[j]=min_sz_t(shape[j],ndshape(args[i])[j]);
    }
  }

  #define CASE(T) {T *p=(T*)param;p[0]=1;p[1]=1; bytesof_param=2*sizeof(T);} break
  TYPECASE(ndtype(x));
  #undef CASE

  #define CASE2(T1,T2) TRY(binary_op(ndim,shape, \
                              nddata(z),ndstrides(z), \
                              nddata(x),ndstrides(x), \
                              nddata(y),ndstrides(y), \
                              (void*)param,bytesof_param, \
                              project_##T1##_##T2)); break
  #define CASE(T)      TYPECASE2(ndtype(z),T); break
  TYPECASE(ndtype(x));
  #undef CASE
  #undef CASE2
  // convert signed vals to unsigned vals in case of type change
  { int b;
    if(b=get_sign_change_bit(ndtype(z),ndtype(x)))
      ndxor_ip(z,1ULL<<b,ndim,shape);
  }
  return z;
Error:
  return NULL;
}

nd_t ndfmad(nd_t z, float a, const nd_t x, float b, const nd_t y,size_t ndim, size_t *shape)
{ nd_t args[] = {z,x,y};
  size_t i;
  float param[] = {a,b};
  TRY(ndkind(x)==ndkind(y));      // Require x and y have the same type, z type may vary
  for(i=0;i<countof(args);++i)
    TRY(ndkind(args[i])==nd_cpu); // this implementation won't work for gpu based arrays
  // set shape and dim if necessary
  if(!ndim)
  { for(i=1,ndim=ndndim(args[0]);i<countof(args);++i)
      ndim=min_sz_t(ndim,ndndim(args[i]));
  }
  if(!shape)
  { TRY(shape=(size_t*)alloca(sizeof(size_t)*ndim));
    memcpy(shape,ndshape(args[0]),sizeof(size_t)*ndim);
    for(i=1;i<countof(args);++i)
    { size_t j;
      for(j=0;j<ndim;++j)
        shape[j]=min_sz_t(shape[j],ndshape(args[i])[j]);
    }
  }
  #define CASE2(T1,T2) TRY(binary_op(ndim,shape, \
                              nddata(z),ndstrides(z), \
                              nddata(x),ndstrides(x), \
                              nddata(y),ndstrides(y), \
                              (void*)param,sizeof(param), \
                              project_##T1##_##T2)); break
  #define CASE(T)      TYPECASE2(ndtype(z),T); break
  TYPECASE(ndtype(x));
  #undef CASE
  #undef CASE2
  // convert signed vals to unsigned vals in case of type change
  { int b;
    if(b=get_sign_change_bit(ndtype(z),ndtype(x)))
      ndxor_ip(z,1ULL<<b,ndim,shape);
  }
  return z;
Error:
  return NULL;
}

#undef LOG
#define LOG(...) ndLogError(z,__VA_ARGS__)
/** In-place xor.
 *  \code
 *  z^=c
 *  \endcode
 *
 *  \param[in,out]  z     The array on which to operate.
 *                        The xor will operate on floating point values via
 *                        integer casting.
 *  \param[in]      c     Each voxel will be xor'd against these bits.
 *  \param[in]      ndim  The number of dimensions in the subvolume on which to
 *                        operate.  If 0, will use the dimensionality of \a z.
 *  \param[in]      shape The shape of the subvolume on which to operate.
 *                        If NULL, will use the full shape of \a z.
 */
nd_t ndxor_ip(nd_t z,uint64_t c,size_t ndim,size_t* shape)
{ size_t i;
  u64 param[] = {c};
  TRY(ndkind(z)==nd_cpu); // this implementation won't work for gpu based arrays
  // set shape and dim if necessary
  if(!ndim)
  { ndim=ndndim(z);
  }
  if(!shape)
  { TRY(shape=(size_t*)alloca(sizeof(size_t)*ndim));
    memcpy(shape,ndshape(z),sizeof(size_t)*ndim);
  }
  #define CASE(T) TRY(inplace_op(ndim,shape, \
                                 nddata(z),ndstrides(z), \
                                 (void*)param,sizeof(param), \
                                 xor_ip_##T)); break
  TYPECASE(ndtype(z));
  #undef CASE
  return z;
Error:
  return NULL;
}

/** In-place voxel type conversion.
 *
 *  In contrast to ndtype(), this performs signed to unsigned interger
 *  mapping.  The mapping preserves the "continuity" of the interval
 *  represented by the type.  It maps the minimum unsigned integer to the
 *  minimum signed integer, and the same for the maximum intergers.
 *
 *  \verabatim
 *  [min signed, max signed] <--> [min unsigned, max unsigned]
 *  \endverbatim
 *
 *  Normal integer casting leave's zero fixed, but this mapping does not.
 *
 *  For other type conversions, this is just a cast.  The cast happens after
 *  the sign change operation.
 *
 *  Since this is in place, the shape of the array may change in response
 *  to the type change.
 *
 *  As an example, when converting to a smaller integer (e.g. i16->u8) the
 *  conversion happens as follows:
 *
 *  1. map i16 to u16.
 *     For example, -4096(0xf000) -> 28672(0x7000)
 *  2. split each u16 voxel into two u8 voxels.
 *     For example, 28672(0x7000) -> 112(0x70),0(0x00)
 *
 *  Reversing the order of operations would give 112(0x70),128(0x80).  I don't
 *  know which answer is best, so I just picked one.
 */
nd_t ndconvert_ip (nd_t z, nd_type_id_t type)
{ int b;
  if(b=get_sign_change_bit(ndtype(z),type))
    ndxor_ip(z,1ULL<<b,0,NULL);
  return ndcast(z,type);
}
