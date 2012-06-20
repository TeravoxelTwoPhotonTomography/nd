/** \file    
    Basic nd-array algorithms 

    \author Nathan Clack
    \date   June 2012

    \todo refactor setting ndim and shape to it's own function
*/
#include <stdint.h>
#include <malloc.h>
#include <stdio.h>
#include "nd.h"

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
#define TRY(e)   do{if(!(e)) {LOG("%s(%d):"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,#e); goto Error; }}while(0)
#define FAIL     do{          LOG("%s(%d):"ENDL "\tExecution should not reach here."ENDL,__FILE__,__LINE__); goto Error; }while(0) 

/////
///// Basic Element-wise operations
/////

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
    TRY(ndkind(x)==nd_cpu);       // this implementation won't work for gpu based arrays
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
    TRY(ndkind(x)==nd_cpu);       // this implementation won't work for gpu based arrays
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
  return z;
Error:
  return NULL;
}