/** \file
 *   Basic nd-array algorithms.
 *
 *   Most of these are designed to operate on sub-volumes of the arrays.
 *
 *   \author Nathan Clack
 *   \date   June 2012
 *
 *   \todo Refactor setting ndim and shape to it's own function.
 */
#include "cuda_runtime_api.h"
#include <stdint.h>
#if defined(__APPLE__) || defined(__MACH__)
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif
#include <stdio.h>
#include "nd.h"
#include "ops.h"

/// @cond DEFINES
#define restrict   __restrict
#define countof(e) (sizeof(e)/sizeof(*(e)))

#ifdef _MSC_VER
#define alloca _alloca
#endif
/// @endcond

/// @cond PRIVATE
// Forward declarations
unsigned xor_ip_cuda(nd_t dst,uint64_t v);
unsigned bitshift_ip_cuda(nd_t dst,int b,int n);

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
/// @endcond

typedef size_t  stride_t;

typedef void (inplace_vec_op_t)(stride_t N, void *z, stride_t zst, void *param, size_t nbytes);                                                   ///< \verbatim 1D f:z=f(z) \endverbatim
typedef void (unary_vec_op_t)(stride_t N,void* z,stride_t zst,const void* x,stride_t xst,void *param, size_t nbytes);                             ///< \verbatim 1D f:z=f(x)   \endverbatim
typedef void (binary_vec_op_t)(stride_t N,void* z,stride_t zst,const void* x,stride_t xst,const void* y,stride_t yst,void *param, size_t nbytes); ///< \verbatim 1D f:z=f(x,y) \endverbatim

//-// Import generics
#include "generic/all.c"
#include "generic/macros.h"

//-// import kind capabilities
#include "private/kind.c"

//-//
//-// Error handling
//-//
/// @cond DEFINES
#define ENDL     "\n"
#define LOG(...) fprintf(stderr,__VA_ARGS__)
#define TRY(e)   do{if(!(e)) {LOG("%s(%d): %s"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error; }}while(0)
#define CUTRY(e) do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,cudaGetErrorString(ecode)); goto Error; }}while(0)
#define TRYMSG(e,msg) do{if(!(e)) {LOG("%s(%d): %s"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,msg); goto Error; }}while(0)
#define FAIL     do{          LOG("%s(%d): %s"ENDL "\tExecution should not reach here."ENDL,__FILE__,__LINE__,__FUNCTION__); goto Error; }while(0)
#define TODO     do{          LOG("%s(%d): %s"ENDL "TODO: \tNot implemented yet."ENDL,__FILE__,__LINE__,__FUNCTION__); exit(-1); }while(0)
/// @endcond

//-//
//-// Basic Element-wise operations
//-//

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

/** The type of the array after a sign change. */
static nd_type_id_t sign_changed_type(nd_type_id_t tid)
{ nd_type_id_t map[]={nd_i8,nd_i16,nd_i32,nd_i64,nd_u8,nd_u16,nd_u32,nd_u64,nd_f32,nd_f64};
  if(tid>nd_id_unknown && tid<nd_id_count) 
    return map[tid];
  return nd_id_unknown;
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

//-// IN-PLACE
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
  int64_t m;
  const stride_t *strides[] = {zst};
  TRY( (m=find_vectorizable_dim(ndim,shape,countof(strides),strides))>=0 );
  inplace_op_recurse(m,ndim-1,shape,z,zst,param,nbytes,f);
  return 1;
Error:
  return 0; // stack overflowed
}

//-// UNARY
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

//-// BINARY
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

//-//
//-// INTERFACE OPS
//-//

static size_t min_sz_t(size_t a, size_t b)
{ return (a<b)?a:b; }

/// @cond DEFINES
#undef LOG
#define LOG(...)
/// @endcond

struct _cuda_copy_param_t
{ enum cudaMemcpyKind direction;
  cudaStream_t stream;
  cudaError_t  ecode; ///< out
};

struct _cuda_copy_param_t make_cuda_copy_params(nd_t dst,nd_t src)
{ struct _cuda_copy_param_t out={cudaMemcpyDeviceToDevice,0,cudaSuccess};
  out.stream=(ndkind(src)==nd_gpu_cuda)?ndCudaStream(src):ndCudaStream(dst);
  if(ndkind(src)==nd_gpu_cuda  && ndkind(dst)!=nd_gpu_cuda)
    out.direction=cudaMemcpyDeviceToHost;
  if(ndkind(src)!=nd_gpu_cuda  && ndkind(dst)==nd_gpu_cuda)
    out.direction=cudaMemcpyHostToDevice;
  return out;
}

void cuda_copy_op(stride_t N,void* z,stride_t zst,const void* x,stride_t xst,void *param, size_t nbytes)
{ struct _cuda_copy_param_t* p=(struct _cuda_copy_param_t*)param;
  cudaError_t ecode=cudaErrorInvalidValue;
  TRY(p->ecode==cudaSuccess);
  TRY(xst==zst);
  CUTRY(ecode=cudaMemcpy(z,x,N*xst,p->direction));
  p->ecode=cudaSuccess;
  return;
Error:
  if(p->ecode!=cudaSuccess) return;
  p->ecode=ecode;
}

/// @cond DEFINES
#undef LOG
#define LOG(...) do{ ndLogError(dst,__VA_ARGS__); ndLogError(src,__VA_ARGS__); } while(0)
/// @endcond

/** Strided copy.
 *
 *  The caller must set up \a dst by allocating memory and make sure it has the
 *  correct type and shape.
 *
 *  This uses the strided order in \src and \dst to determine which elements
 *  get copied where.
 *
 *  \param[in,out]  dst   The output array.
 *  \param[in]      src   The input array.
 *  \param[in]     ndim   The number of dimensions in the sub-volume described by \a shape.
 *                        If 0, this is set to the largest dimension that still fits \a src
 *                        and \a dst.
 *  \param[in]    shape   The copy is restricted to a sub-volume with this shape.
 *                        If NULL, the smallest shape common to \a src and \a dst will
 *                        be used.
 *  \returns \a dst on success, or NULL otherwise.
 *  \ingroup ndops
 */
nd_t ndcopy(nd_t dst, const nd_t src, size_t ndim, size_t *shape)
{ REQUIRE(src,PTR_ARITHMETIC);
  REQUIRE(dst,PTR_ARITHMETIC);
  // if shape or ndims is unspecified use smallest
  if(!ndim)
    ndim=min_sz_t(ndndim(dst),ndndim(src));
  if(!shape)
  { size_t i;
    TRY(shape=(size_t*)alloca(sizeof(size_t)*ndim));
    for(i=0;i<ndim;++i)
      shape[i]=min_sz_t(ndshape(dst)[i],ndshape(src)[i]);
  }
  if(ndkind(src)==nd_gpu_cuda || ndkind(dst)==nd_gpu_cuda)
  { struct _cuda_copy_param_t param=make_cuda_copy_params(dst,src);
    TRY(unary_op(ndim,shape,nddata(dst),ndstrides(dst),nddata(src),ndstrides(src),&param,sizeof(param),cuda_copy_op));
    CUTRY(param.ecode);
  } else
  { // RAM
    REQUIRE(src,CAN_MEMCPY);
    REQUIRE(dst,CAN_MEMCPY);
    /// @cond DEFINES
    #define CASE2(T1,T2) TRY(unary_op(ndim,shape, \
                                nddata(dst),ndstrides(dst), \
                                nddata(src),ndstrides(src), \
                                0,0, \
                                copy_##T1##_##T2)); break
    #define CASE(T)      TYPECASE2(ndtype(dst),T); break
    /// @endcond
        TYPECASE(ndtype(src));
    #undef CASE
    #undef CASE2
    // convert signed vals to unsigned vals in case of type change
    { int b;
      if((b=get_sign_change_bit(ndtype(dst),ndtype(src))))
        ndxor_ip(dst,1ULL<<b,ndim,shape);
    }
  }
  return dst;
Error:
  return NULL;
}

/** Transpose two dimensions \a i and \a j.
 *
 *  The caller must set up \a dst by allocating memory and make sure it has the
 *  correct type and shape.  This operation acts as if the data was first copied
 *  from \a src to \a dst (as in ndcopy()), and then the \a dst array is transposed
 *  in place.
 *
 *  In reality, the transpose is accomplised during the copy by manipulating the
 *  shape and strides of \a dst.
 *
 *  Example:
 *  \code{c}
 *  // transpose vol
 *  { nd_t dst=ndheap(vol);
 *    EXPECT_EQ(dst,ndtranspose(dst,vol,2,3,0,NULL));
 *    ndfree(vol); // replace original volume with transposed version
 *    vol=dst; 
 *  }
 *  \endcode
 *
 *  \param[in,out]  dst   The output array.  The shapes for dimensions i and j will be switched.
 *  \param[in]      src   The input array.
 *  \param[in]     ndim   The number of dimensions in the sub-volume described by \a shape.
 *                        If 0, this is set to the largest dimension that still fits \a src
 *                        and \a dst.
 *  \param[in]    shape   The copy is restricted to a sub-volume with this shape.
 *                        If NULL, the smallest shape common to \a src and \a dst will
 *                        be used.  The shape should be specified in the source space
 *                        (before transposition).
 *  \param[in]        i   This dimension will be switched with dimension \a j.
 *  \param[in]        j   This dimension will be switched with dimension \a i.
 *  
 *  \returns \a dst on success, or NULL otherwise.
 *  \ingroup ndops
 */

nd_t ndtranspose(nd_t dst, const nd_t src, unsigned i, unsigned j, size_t ndim, size_t *shape)
{ size_t ti,tj;
  unsigned idim;

  // Compute domain if left NULL
  if(!ndim)
    ndim=min_sz_t(ndndim(dst),ndndim(src));
  if(!shape) // use the source shape by default
  { shape=(size_t*)alloca(sizeof(size_t)*ndim);
    for(idim=0;idim<ndim;++idim)
      shape[idim]=min_sz_t(ndshape(dst)[idim],ndshape(src)[idim]);
  }

  // permute dst shape
  { size_t t=ndshape(dst)[i];
    ndShapeSet(dst,i,ndshape(dst)[j]);
    ndShapeSet(dst,j,t);
    ti=ndstrides(dst)[i],
    tj=ndstrides(dst)[j];
  }

  // Transpose by copying with permuted pitches
  ndstrides(dst)[i]=tj;
  ndstrides(dst)[j]=ti;
  ndcopy(dst,src,ndim,shape);
  ndstrides(dst)[i]=ti;
  ndstrides(dst)[j]=tj;
  return dst;
}

/// @cond DEFINES
#undef LOG
#define LOG(...) do{ ndLogError(dst,__VA_ARGS__); } while(0)
/// @endcond

/**
 *  Circularly shift the dimensions of \a src by \a n and place the result in \a dst. 
 *
 *  The caller must set up \a dst by allocating memory and make sure it has the
 *  correct type and shape.  This operation acts as if the data was first copied
 *  from \a src to \a dst (as in ndcopy()), and then the \a dst array is operated on
 *  in place.
 *
 *  In reality, this is accomplished during the copy by manipulating the shape and 
 *  strides of \a dst.
 *
 *  Example:
 *  \code{c}
 *  { nd_t dst=ndheap(vol);
 *    EXPECT_EQ(dst,ndshiftdim(dst,vol,3));
 *    ndfree(vol); // replace original volume with transposed version
 *    vol=dst; 
 *  }
 *  \endcode
 *
 *  \param[in,out]  dst   The output array.
 *  \param[in]      src   The input array.
 *  \param[in]        n   The number of places to shift the dimension.  Positive n shifts dimensions
 *                        to the right (e.g. n=1 will move dimension 0 to 1, 1 to 2, and so on).
 *  
 *  \returns \a dst on success, or NULL otherwise.
 *  \ingroup ndops
 * 
 *  \todo handle when \a src and \a dst arrays with different numbers of dimensions? 
 *        Have different shape?
 *        At the moment, \a dst is treated as a bag of bits and that's about it.
 */
nd_t ndshiftdim(nd_t dst,const nd_t src,int n)
{ size_t i,imap[32]={0},map[32]={0},s[32]={0};
  const unsigned d=ndndim(src);
  TRY(d<countof(map));
  //TRY(ndndim(dst)==ndndim(src));  // Don't need to check because we'll add needed dimensions, and we don't care about extras
  for(i=0;i<d;++i)  map[(i+n)%d]=i; // compute the permutation
  for(i=0;i<d;++i) imap[(i-n)%d]=i; // compute the inverse permutation
  
  for(i=0;i<d;++i) s[i]=ndshape(dst)[i];        // hold the shape temporarily
  for(i=0;i<d;++i) ndShapeSet(dst,i,s[map[i]]); // permute dst shape
  
  // Copy with permuted pitches
  for(i=0;i<d;++i) s[i]=ndstrides(dst)[i];      // hold the strides temporarily
  for(i=0;i<d;++i) ndstrides(dst)[i]=s[imap[i]];// permute strides
  ndcopy(dst,src,d,ndshape(src));
  for(i=0;i<d;++i) ndstrides(dst)[i]=s[i];      // restore strides

  return dst;
Error:
  return 0;
}

/// @cond DEFINES
#undef LOG
#define LOG(...) do{ ndLogError(z,__VA_ARGS__); ndLogError(x,__VA_ARGS__); ndLogError(y,__VA_ARGS__);} while(0)
/// @endcond
/** Add two arrays.
 * \code
 *    z=x+y
 * \endcode
 * Operations are performed in the source array type and then cast to the destination type.
 * \param[in,out]    z    The output array.
 * \param[in]        x    An input array.
 * \param[in]        y    An input array.
 * \param[in]     ndim    The number of dimensions in the sub-volume described by \a shape.
 *                        If 0, this is set to the largest dimension that still fits \a x, \a y, and \a z.
 * \param[in]    shape    Computation is restricted to a sub-volume with this shape.
 *                        If NULL, the smallest shape common to \a x, \a y, and \a z will
 *                        be used.
 *  \returns \a z on success, or NULL otherwise.
 * \ingroup ndops
 */
nd_t ndadd(nd_t z, const nd_t x, const nd_t y, size_t ndim, size_t *shape)
{ nd_t args[] = {z,(nd_t)x,(nd_t)y};
  size_t i, bytesof_param;
  u8 param[8*2];
  TRY(ndkind(x)==ndkind(y));      // Require x and y have the same type, z type may vary
  for(i=0;i<countof(args);++i)
    REQUIRE(args[i],PTR_ARITHMETIC|CAN_MEMCPY);
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

  /// @cond DEFINES
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
  /// @endcond
  // convert signed vals to unsigned vals in case of type change
  { int b;
    if((b=get_sign_change_bit(ndtype(z),ndtype(x))))
      ndxor_ip(z,1ULL<<b,ndim,shape);
  }
  return z;
Error:
  return NULL;
}

/** Floating multiply and add.
 *  \code
 *    z = a*x+b*y
 *  \endcode
 *  Operations are performed in floating-point and then cast to the destination type.
 *
 *  \param[out]    z   The output array.
 *  \param[in]     a   The first coefficient.
 *  \param[in]     x   The first input array.
 *  \param[in]     b   The second coefficient.
 *  \param[in]     y   The first input array.
 *  \param[in]  ndim   The number of dimensions in the sub-volume described by \a shape.
 *                     If 0, this is set to the largest dimension that still fits \a x,
 *                     \a y, and \a z.
 *  \param[in] shape   Computation is restricted to a sub-volume with this shape.
 *                     If NULL, the smallest shape common to \a x, \a y, and \a z will
 *                     be used.
 *  \returns \a z on success, or NULL otherwise.
 *  \ingroup ndops
 */
nd_t ndfmad(nd_t z, float a, const nd_t x, float b, const nd_t y,size_t ndim, size_t *shape)
{ nd_t args[] = {z,x,y};
  size_t i;
  float param[] = {a,b};
  TRY(ndtype(x)==ndtype(y));      // Require x and y have the same type, z type may vary
  for(i=0;i<countof(args);++i)
    REQUIRE(args[i],PTR_ARITHMETIC|CAN_MEMCPY);
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
  /// @cond DEFINES
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
  /// @endcond
  // convert signed vals to unsigned vals in case of type change
  { int b;
    if((b=get_sign_change_bit(ndtype(z),ndtype(x))))
      ndxor_ip(z,1ULL<<b,ndim,shape);
  }
  return z;
Error:
  return NULL;
}

/// @cond DEFINES
#undef LOG
#define LOG(...) ndLogError(z,__VA_ARGS__)
/// @endcond

extern unsigned fill_cuda(nd_t dst,uint64_t v);
/** Set each voxel in \a z to \a c (reinterpreted as ndtype(z))
 *  \code
 *  ndcast(z,nd_f64);          // z is an array of doubles.
 *  ndfill(z,*(int64_t*)&8.5); // fill z with the value 8.5
 *  \endcode
 *
 *  \param[in,out]  z The array on which to operate.
 *  \param[in]      c Each voxel will be set to \a c through a cast.
 *  \returns \a z on success, or NULL otherwise.
 *  \ingroup ndops
 */
nd_t ndfill(nd_t z,uint64_t c)
{ u64 param[] = {c};
  REQUIRE(z,PTR_ARITHMETIC);
  if(ndkind(z)==nd_gpu_cuda)
  { TRY(fill_cuda(z,c));
  } else
  { REQUIRE(z,CAN_MEMCPY);
    /// @cond DEFINES
    #define CASE(T) TRY(inplace_op(ndndim(z),ndshape(z), \
                                   nddata(z),ndstrides(z), \
                                   (void*)param,sizeof(param), \
                                   fill_ip_##T)); break
    /// @endcond
    TYPECASE(ndtype(z));  
    #undef CASE
  }
  return z;
Error:
  return NULL;
}

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
 *  \returns \a z on success, or NULL otherwise.
 *  \ingroup ndops
 */
nd_t ndxor_ip(nd_t z,uint64_t c,size_t ndim,size_t* shape)
{ u64 param[] = {c};
  REQUIRE(z,PTR_ARITHMETIC);
  // set shape and dim if necessary
  if(!ndim)
  { ndim=ndndim(z);
  }
  if(!shape)
  { TRY(shape=(size_t*)alloca(sizeof(size_t)*ndim));
    memcpy(shape,ndshape(z),sizeof(size_t)*ndim);
  }
  if(ndkind(z)==nd_gpu_cuda)
  { TRY(xor_ip_cuda(z,c));
  } else
  { REQUIRE(z,CAN_MEMCPY);
    /// @cond DEFINES
    #define CASE(T) TRY(inplace_op(ndim,shape, \
                                   nddata(z),ndstrides(z), \
                                   (void*)param,sizeof(param), \
                                   xor_ip_##T)); break
    /// @endcond
    TYPECASE(ndtype(z));
    #undef CASE
  }
  return z;
Error:
  return NULL;
}

/** In-place bitshift.
 *  \code
 *  z = (z<<b)&m
 *  \endcode
 *  where \c b=bits and \c m=~(1<<overflow_bit).
 *  \endcode
 *
 *  \param[in,out]  z       The array on which to operate.
 *                          The xor will operate on floating point values via
 *                          integer casting.
 *  \param[in]      bits    The number of bits to shift.  Positive is a left shift.
 *                          Negative values give a right shift.
 *  \param[in] overflow_bit Bits at greater than the overflow bit get masked out.
 *  \returns \a z on success, or NULL otherwise.
 *  \ingroup ndops
 */
nd_t ndbitshift_ip(nd_t z,int bits,unsigned overflow_bit)
{ REQUIRE(z,PTR_ARITHMETIC);
  if(ndkind(z)==nd_gpu_cuda)
  { TRY(bitshift_ip_cuda(z,bits,overflow_bit));
  } else
  { int param[] = {bits,(int)overflow_bit};
    REQUIRE(z,CAN_MEMCPY);
    /// @cond DEFINES
    #define CASE(T) TRY(inplace_op(ndndim(z),ndshape(z), \
                                   nddata(z),ndstrides(z), \
                                   (void*)param,sizeof(param), \
                                   bitshift_ip_##T)); break
    /// @endcond
    TYPECASE_INTEGERS(ndtype(z));
    #undef CASE
  }
  return z;
Error:
  return NULL;
}

/** 
 *  In-place voxel type conversion.
 *
 *  In contrast to ndcast(), this performs signed to unsigned interger
 *  mapping.  The mapping preserves the "continuity" of the interval
 *  represented by the type.  It maps the minimum unsigned integer to the
 *  minimum signed integer, and the same for the maximum integers.
 *
 *  \verbatim
      [min signed, max signed] <--> [min unsigned, max unsigned]
    \endverbatim
 *
 *  Normal integer casting leaves zero fixed, but this mapping does not.
 *
 *  For other type conversions, this is just a cast.  The cast happens after
 *  the sign change operation.
 *
 *  Since this is in place, the shape of the array may change in response
 *  to the type change.
 *
 *  As an example, when converting to a smaller integer (e.g. \c i16->u8) the
 *  conversion happens as follows:
 *
 *  1. map \c i16 to \c u16.
 *     For example, <tt> -4096(0xf000) -> 28672(0x7000) </tt>
 *  2. split each \c u16 voxel into two \c u8 voxels.
 *     For example, <tt> 28672(0x7000) -> 112(0x70),0(0x00) </tt>
 *
 *  Reversing the order of operations (split bytes, then sign change)
 *  would give <tt>112(0x70),128(0x80)</tt>.  I don't
 *  know which answer is best, so I just picked one.
 *
 *  \returns \a z on success, or NULL otherwise.
 *  \ingroup ndops
 */
static size_t ndtype_bpp(nd_type_id_t type)
{
  static const size_t _Bpp[]={1,2,4,8,1,2,4,8,4,8};
  if(type<=nd_id_unknown) return 0;
  if(type>=nd_id_count)   return 0;
  return _Bpp[(unsigned)type];
}

#define _isfloating(tid) ((tid)>nd_i64)

nd_t ndconvert_ip (nd_t z, nd_type_id_t type)
{ int b;
  if((b=get_sign_change_bit(type,ndtype(z))))
    ndcast(ndxor_ip(z,1ULL<<b,0,NULL),sign_changed_type(ndtype(z)));
  
  if((b=ndtype_bpp(type)-ndbpp(z))!=0)
  { nd_t t;
    nd_type_id_t zt=ndtype(z);
    TRY(t=ndmake(ndcast(z,type)));
    ndcast(z,zt);
    if(b<0 && !_isfloating(type) && !_isfloating(ndtype(z)) )
      TRY(ndbitshift_ip(z,8*b,8*ndtype_bpp(type)));
    TRY(ndcopy(t,z,0,0));
    ndswap(t,z);
    ndfree(t);
  }
  else
  { TRY(ndcast(z,type));
  }

  return z;
Error:
  return 0;
}
