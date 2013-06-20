/**
 * \file Helpers for elemnt-wise operations 
 *
 * Include this once in source files where it is needed.  Never include this
 * in a public header.
 *
 * Requires several definitions/declarations:
  *  - TRY(e)
 *   - stride_t
 *   - size_t
 */

#ifdef _MSC_VER
#include <malloc.h>
#define alloca _alloca
#endif

typedef void (inplace_vec_op_t)(stride_t N, void *z, stride_t zst, void *param, size_t nbytes);                                                   ///< \verbatim 1D f:z=f(z) \endverbatim
typedef void (unary_vec_op_t)(stride_t N,void* z,stride_t zst,const void* x,stride_t xst,void *param, size_t nbytes);                             ///< \verbatim 1D f:z=f(x)   \endverbatim
typedef void (binary_vec_op_t)(stride_t N,void* z,stride_t zst,const void* x,stride_t xst,const void* y,stride_t yst,void *param, size_t nbytes); ///< \verbatim 1D f:z=f(x,y) \endverbatim
typedef void (ternary_vec_op_t)(stride_t N,void* z,stride_t zst,const void* x,stride_t xst,const void* y,stride_t yst,const void *w, stride_t wst, void *param, size_t nbytes); ///< \verbatim 1D f:z=f(x,y,w) \endverbatim
typedef void (many_vec_op_t)(stride_t N, void *z, stride_t zst, const size_t narg, const void** args, const stride_t *strides, void *param, size_t nbytes); ///< \verbatim 1D f:z=f(*args) \endverbatim


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

//-// TERNARY
static void ternary_op_recurse(
    int64_t m,
    int64_t idim,
    const stride_t *shape,
          void* z,const stride_t *zst,
    const void* x,const stride_t *xst,
    const void* y,const stride_t *yst,
    const void* w,const stride_t *wst,
    void* param, size_t nbytes,
    ternary_vec_op_t *f)
{
  if(idim<m)
  { f(zst[m]/zst[0],z,zst[0],x,xst[0],y,yst[0],w,wst[0],param,nbytes);
    return;
  }
  if(idim<1)
  { f(shape[0],z,zst[0],x,xst[0],y,yst[0],w,wst[0],param,nbytes);
    return;
  }
  { stride_t i;
    const stride_t ox = xst[idim],
                   oy = yst[idim],
                   oz = zst[idim],
                   ow = wst[idim];
    for(i=0;i<shape[idim];++i)
      ternary_op_recurse(m,idim-1,shape,(u8*)z+oz*i,zst,(u8*)x+ox*i,xst,(u8*)y+oy*i,yst,(u8*)w+ow*i,wst,param,nbytes,f);
  }
}

static int ternary_op(
    stride_t ndim,
    stride_t *shape,
          void* z,const stride_t *zst,
    const void* x,const stride_t *xst,
    const void* y,const stride_t *yst,
    const void* w,const stride_t *wst,
    void* param, size_t nbytes,
    ternary_vec_op_t *f)
{
  int64_t m;
  const stride_t *strides[] = {zst,xst,yst,wst};
  TRY( (m=find_vectorizable_dim(ndim,shape,countof(strides),strides))>=0 );
  ternary_op_recurse(m,ndim-1,shape,z,zst,x,xst,y,yst,w,wst,param,nbytes,f);
  return 1;
Error:
  return 0; // stack overflowed
}

//-// MANY
//typedef void (many_vec_op_t)(stride_t N, void *z, stride_t zst, const size_t narg, const void** args, const stride_t *strides, void *param, size_t nbytes); ///< \verbatim 1D f:z=f(*args) \endverbatim
static void many_op_recurse(
    int64_t m,
    int64_t idim,
    const stride_t *shape,
          void* z,const stride_t* zst,
    const size_t narg,
    const void** args,const stride_t **strides,
    void* param, size_t nbytes,
    many_vec_op_t *f)
{ size_t i;
  if(idim<m || idim<1)
  { stride_t *st=(stride_t*)alloca(sizeof(stride_t)*narg);
    for(i=0;i<narg;++i)
      st[i]=strides[i][0];
    if(idim<m)
    { f(zst[m]/zst[0],z,zst[0],narg,args,st,param,nbytes);
      return;
    }
    if(idim<1)
    { f(shape[0],z,zst[0],narg,args,st,param,nbytes);
      return;
    }
  }
  { stride_t i;
    size_t iarg;
    u8** a=(u8**)args;
    for(i=0;i<shape[idim];++i)
    { const stride_t oz = zst[idim];
      many_op_recurse(m,idim-1,shape,(u8*)z+oz*i,zst,narg,args,strides,param,nbytes,f); // (u8*)x+ox*i,xst,(u8*)y+oy*i,yst,(u8*)w+ow*i,wst
      for(iarg=0;iarg<narg;++iarg)
        a[iarg]+=strides[iarg][idim];
    }
    for(iarg=0;iarg<narg;++iarg)
      a[iarg]-=shape[idim]*strides[iarg][idim];
  }
}

static int many_op(
    stride_t ndim,
    stride_t *shape,
          void* z,const stride_t *zst,
    const size_t narg,
    const void** args,const stride_t **strides,
    void* param, size_t nbytes,
    many_vec_op_t *f)
{
  int64_t m;
  TRY( (m=(int64_t)find_vectorizable_dim(ndim,shape,(int)narg,strides))>=0 );
  many_op_recurse(m,ndim-1,shape,z,zst,narg,args,strides,param,nbytes,f);
  return 1;
Error:
  return 0; // stack overflowed
}
