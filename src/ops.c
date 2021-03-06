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
#include "config.h"
#if HAVE_CUDA
  #include "cuda_runtime_api.h"
#endif
#include <stdint.h>
#include <limits.h>
#if defined(__APPLE__) || defined(__MACH__)
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif
#include <stdio.h>
#include <float.h>
#include <stdarg.h>
#include "nd.h"
#include "ops.h"

// It looks like older versions of gcc don't define these
# ifndef LLONG_MIN
#  define LLONG_MIN	(-LLONG_MAX-1)
# endif
# ifndef LLONG_MAX
#  define LLONG_MAX	__LONG_LONG_MAX__
# endif
# ifndef ULLONG_MAX
#  define ULLONG_MAX	(LLONG_MAX * 2ULL + 1)
# endif

/// @cond DEFINES
#define restrict   __restrict
#define countof(e) (sizeof(e)/sizeof(*(e)))

#ifdef _MSC_VER
#define alloca _alloca
#endif
/// @endcond

/// @cond PRIVATE
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

typedef union val_t_ { int d; unsigned u; unsigned long long llu; long long lld; double f;} val_t; ///< Generic value type for passing parameters.
#define VAL_u8(v)  (v.u)
#define VAL_u16(v) (v.u)
#define VAL_u32(v) (v.u)
#define VAL_u64(v) (v.llu)
#define VAL_i8(v)  (v.d)
#define VAL_i16(v) (v.d)
#define VAL_i32(v) (v.d)
#define VAL_i64(v) (v.lld)
#define VAL_f32(v) (v.f)
#define VAL_f64(v) (v.f)
#define VAL_(type) VAL_##type
#define VAL(v,type) VAL_(type)(v)

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
#define max_f64  DBL_MAX


//-//
//-// Error handling
//-//

/// @cond DEFINES
static void breakme() {}
#define ENDL     "\n"
#define LOG(...) fprintf(stderr,__VA_ARGS__)
#define TRY(e)   do{if(!(e)) {LOG("%s(%d): %s"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); breakme(); goto Error; }}while(0)
#if HAVE_CUDA
  #define CUTRY(e) do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,cudaGetErrorString(ecode)); goto Error; }}while(0)
#else
  #define CUTRY TRY
#endif
#define TRYMSG(e,msg) do{if(!(e)) {LOG("%s(%d): %s"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,msg); goto Error; }}while(0)
#define FAIL     do{          LOG("%s(%d): %s"ENDL "\tExecution should not reach here."ENDL,__FILE__,__LINE__,__FUNCTION__); goto Error; }while(0)
#define TODO     do{          LOG("%s(%d): %s"ENDL "TODO: \tNot implemented yet."ENDL,__FILE__,__LINE__,__FUNCTION__); exit(-1); }while(0)
/// @endcond

//-// Import generics
#include "generic/all.c"
#include "generic/macros.h"

//-// import kind capabilities
#include "private/kind.c"
#include "private/element-wise.h"

#if HAVE_CUDA
extern unsigned xor_ip_cuda(nd_t dst,uint64_t v);
extern unsigned bitshift_ip_cuda(nd_t dst,int b,int n);
extern unsigned saturate_ip_cuda(nd_t dst,val_t mn, val_t mx);
extern unsigned fill_cuda(nd_t dst,uint64_t v);
extern unsigned fmad_scalar_ip_cuda(nd_t dst,float m, float b);
extern unsigned fmad_cuda(nd_t z,nd_t a,nd_t x,nd_t b,size_t ndim,size_t *shape);
#else
#define NO_CUDA_SUPPORT {FAIL; Error: return 0;}
unsigned xor_ip_cuda(nd_t dst,uint64_t v)                                 NO_CUDA_SUPPORT
unsigned bitshift_ip_cuda(nd_t dst,int b,int n)                           NO_CUDA_SUPPORT
unsigned saturate_ip_cuda(nd_t dst,val_t mn, val_t mx)                    NO_CUDA_SUPPORT
unsigned fill_cuda(nd_t dst,uint64_t v)                                   NO_CUDA_SUPPORT
unsigned fmad_scalar_ip_cuda(nd_t dst,float m, float b)                   NO_CUDA_SUPPORT
unsigned fmad_cuda(nd_t z,nd_t a,nd_t x,nd_t b,size_t ndim,size_t *shape) NO_CUDA_SUPPORT
#endif

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
{ 
#if HAVE_CUDA
  enum cudaMemcpyKind direction;
  cudaStream_t stream;
  cudaError_t  ecode; ///< out
#else
  int ecode;
#endif
};

#if HAVE_CUDA
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
#else
  struct _cuda_copy_param_t make_cuda_copy_params(nd_t dst,nd_t src)
  { struct _cuda_copy_param_t out={0};
    return out;
  }
  void cuda_copy_op(stride_t N,void* z,stride_t zst,const void* x,stride_t xst,void *param, size_t nbytes)  {return;}
#endif

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
  TRY(!nderror(dst));
  TRY(!nderror(src));
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
  //TRY(ndndim(dst)==ndndim(src));    // Don't need to check because we'll add needed dimensions, and we don't care about extras
  for(i=0;i<d;++i)  map[(i+n)%d]=i;   // compute the permutation
  for(i=0;i<d;++i) imap[(i+d-n)%d]=i; // compute the inverse permutation
  
  for(i=0;i<d;++i) s[i]=ndshape(dst)[i];        // hold the shape temporarily
  for(i=0;i<d;++i) ndShapeSet(dst,(int)i,s[map[i]]); // permute dst shape
  
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


/// @cond DEFINES
#undef LOG
#define LOG(...) ndLogError(z,__VA_ARGS__)
/// @endcond
//
/** Floating multiply and add.
 *  \code
 *    z = a.*x+b
 *  \endcode
 *  Operations are performed in floating-point and then cast to the destination type
 *
 *  \param[out]    z   The output array.
 *  \param[in]     a   An array.
 *  \param[in]     x   An array.
 *  \param[in]     b   An array.  If NULL, it will be ignored.
 *  \param[in]  ndim   The number of dimensions in the sub-volume described by \a shape.
 *                     If 0, this is set to the largest dimension that still fits \a x,
 *                     \a y, and \a z.
 *  \param[in] shape   Computation is restricted to a sub-volume with this shape.
 *                     If NULL, the smallest shape common to \a x, \a y, and \a z will
 *                     be used.
 *  \returns \a z on success, or NULL otherwise.
 *  \ingroup ndops
 */
nd_t ndfmad(nd_t z,const nd_t a,const nd_t x,const nd_t b,size_t ndim,size_t* shape)
{ nd_t args[] = {z,a,x,b};
  size_t i;
  // Check preconditions on input
  for(i=0;i<countof(args);++i)
  { TRY(args[i]);
    REQUIRE(args[i],PTR_ARITHMETIC);
    TRY(ndkind(z)==ndkind(args[i]));
  }
  TRY(ndtype(a)==ndtype(x));      // Require source arrays to have the same type, z type may vary
  TRY(ndtype(a)==ndtype(b));
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
  if(ndkind(z)==nd_gpu_cuda)
  { TRY(fmad_cuda(z,a,x,b,0,0));
  } else
  { /// @cond DEFINES
    #define CASE2(T1,T2) TRY(ternary_op(ndim,shape, \
                                nddata(z),ndstrides(z), \
                                nddata(a),ndstrides(a), \
                                nddata(x),ndstrides(x), \
                                nddata(b),ndstrides(b), \
                                NULL,0, \
                                fmad_##T1##_##T2)); break
    #define CASE(T)      TYPECASE2(ndtype(z),T); break
    TYPECASE(ndtype(x));
    #undef CASE
    #undef CASE2
  }
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

/** Scalar floating multiply and add (in place, with saturation). 
 *  \code
 *    z = m*z+b
 *  \endcode
 *  Operations are performed in floating-point and then cast to the destination type.
 *
 *  \param[out]    z   The output array.
 *  \param[in]     m   The slope.
 *  \param[in]     b   The intercept.
 *  \param[in]  ndim   The number of dimensions in the sub-volume described by \a shape.
 *                     If 0, this is set to the largest dimension that still fits \a x,
 *                     \a y, and \a z.
 *  \param[in] shape   Computation is restricted to a sub-volume with this shape.
 *                     If NULL, the smallest shape common to \a x, \a y, and \a z will
 *                     be used.
 *  \returns \a z on success, or NULL otherwise.
 *  \ingroup ndops
 */
nd_t ndfmad_scalar_ip(nd_t z,float m,float b,size_t ndim,size_t *shape)
{ f32 param[] = {m,b};
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
  { TRY(fmad_scalar_ip_cuda(z,m,b));
  } else
  { REQUIRE(z,CAN_MEMCPY);
    /// @cond DEFINES
    #define CASE(T) TRY(inplace_op(ndim,shape, \
                                   nddata(z),ndstrides(z), \
                                   (void*)param,sizeof(param), \
                                   fmad_scalar_ip_##T)); break
    /// @endcond
    TYPECASE(ndtype(z));
    #undef CASE
  }
  return z;
Error:
  return NULL;
}

/// @cond DEFINES
#undef LOG
#define LOG(...) ndLogError(z,__VA_ARGS__)
/// @endcond

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
 *  \todo This is a terrible function.  It does not behave intuitively.  FIX. 
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

/** In-place linear contrast adjustment with satuation.
 *
 *  Determines the linear transform needed to map \a min and \a max to interval 
 *  representable by ndtype(z).
 *
 *  For example to map part of an \c int16 range to 0-255 in a \c uint8 array:
 *  \code
 *  ndLinearConstrastAdjust_ip(z,nd_u8,-10000,4000); // maps -10000 to 0 and 4000 to 255.
 *  ndconvert_ip(z,nd_u8);
 *  \endcode
 *  
 *  The linear scaling exhibits type saturation.  In the example, everything 
 *  with an intensity less than -10000 in the original also gets mapped to 0.
 *  Everything brighter than 4000 gets mapped to 255.
 *
 *  The va_arg() mechanism is used to pass the \a min and \a max arguments in a
 *  generic fashion.  The type of min and max are infered from ndtype(z).  
 *  This is somewhat dangerous if you try passing mismatched types.
 *
 *  For floating point types, the maximum and minimum of the output range are
 *  0 and 1.
 * 
 *  \see ndfmad_scalar_ip()
 *
 *  \param[out]  z   The array on which to operate.
 *  \param[in]  type The type from which to infer the output range.
 *  \param[in]   min The minimum of the input intensity range. The type should match the value type of \a z.
 *  \param[in]   max The maximum of the input intensity range. The type should match the value type of \a z.
 *  \returns \a z on success, or NULL otherwise.
 *  \ingroup ndops
 */
nd_t ndLinearConstrastAdjust_ip(nd_t z,nd_type_id_t dtype,.../*min,max*/)
{ float mn,mx,m,b;
  #define TMIN(T) min_##T
  #define TMAX(T) max_##T
  const float mns[]=
  { 0.0,0.0,0.0,0.0, //nd_u8,  nd_u16,  nd_u32,  nd_u64,
    (f32)TMIN(i8),(f32)TMIN(i16),(f32)TMIN(i32),(f32)TMIN(i64),
    0.0,0.0 //f32,f64
  };
  const float mxs[]=
  { (f32)TMAX(u8),(f32)TMAX(u16),(f32)TMAX(u32),(f32)TMAX(u64),
    (f32)TMAX(i8),(f32)TMAX(i16),(f32)TMAX(i32),(f32)TMAX(i64),
    1.0,1.0 //f32,f64
  };
  #undef TMIN
  #undef TMAX

  { va_list ap;
    va_start(ap,dtype);
    switch(ndtype(z))
    { case nd_u8:
      case nd_u16:
      case nd_i8:
      case nd_i16:
      case nd_i32: {mn=(f32)va_arg(ap,int); mx=(f32)va_arg(ap,int);} break;
      case nd_u32: {mn=(f32)va_arg(ap,unsigned); mx=(f32)va_arg(ap,unsigned);} break;
      case nd_f32:
      case nd_f64: {mn=(f32)va_arg(ap,double); mx=(f32)va_arg(ap,double);} break;
      default:FAIL; /// \todo handle i64 and u64
    }
    va_end(ap);
  }
  m=(mxs[dtype]-mns[dtype])/(mx-mn);
  b=-m*mn;
  return ndfmad_scalar_ip(z,m,b,ndndim(z),ndshape(z));
Error:
  return 0;
}

/** In-place satuation.
 *
 *  Intensities are inclusively clamped to the interval from \a min to \a max:
 *  Values less than \a min are set to \a min, while values more than \a max are
 *  set to \a max.
 *
 *  The va_arg() mechanism is used to pass the \a min and \a max arguments in a
 *  generic fashion.  The type of min and max are infered from ndtype(z).  
 *  This is somewhat dangerous if you try passing mismatched types.
 *
 *  \param[out]  z   The array on which to operate.
 *  \param[in]   min The minimum of the input intensity range. Type should match the pixel type of \a z.
 *  \param[in]   max The maximum of the input intensity range. Type should match the pixel type of \a z.
 *  \returns \a z on success, or NULL otherwise.
 *  \ingroup ndops
 */
nd_t ndsaturate_ip   (nd_t z,.../*min,max*/)
{ val_t mn={0},mx={0}; // It's important to init these for some compilers (msvc)
  { va_list ap;
    va_start(ap,z);
    switch(ndtype(z))
    { case nd_u8:
      case nd_u16:
      case nd_i8:
      case nd_i16:
      case nd_i32: {mn.d  =va_arg(ap,int);                mx.d  =va_arg(ap,int);}                break;
      case nd_u32: {mn.u  =va_arg(ap,unsigned);           mx.u  =va_arg(ap,unsigned);}           break;
      case nd_i64: {mn.lld=va_arg(ap,long long);          mx.lld=va_arg(ap,long long);}          break;
      case nd_u64: {mn.llu=va_arg(ap,unsigned long long); mx.llu=va_arg(ap,unsigned long long);} break;
      case nd_f32:
      case nd_f64: {mn.f  =va_arg(ap,double);   mx.f  =va_arg(ap,double);}   break;
      default:FAIL; // could not understand type
    }
    va_end(ap);
  }
  { val_t param[2]; // = {mn,mx}; -- for some reason this doesn't work on msvc...may be missing something, but I think msvc doesn't treat these as plain old data
    param[0]=mn;
    param[1]=mx;
    REQUIRE(z,PTR_ARITHMETIC);
    if(ndkind(z)==nd_gpu_cuda)
    { TRY(saturate_ip_cuda(z,mn,mx));
    } else
    { REQUIRE(z,CAN_MEMCPY);
      /// @cond DEFINES
      #define CASE(T) TRY(inplace_op(ndndim(z),ndshape(z), \
                                     nddata(z),ndstrides(z), \
                                     (void*)param,sizeof(param), \
                                     saturate_ip_##T)); break
      /// @endcond
      TYPECASE(ndtype(z));
      #undef CASE
    }
  }
  return z;
Error:
  return 0;
}
