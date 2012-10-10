/** \file
 *  Implementation for concatenation operators.
 */
#include "nd.h"
#include "ops.h"

#include "private/kind.c"

/// @cond DEFINES
#define ENDL     "\n"
#define TRY(e)   do{if(!(e)) {LOG("%s(%d): %s"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error; }}while(0)
#define TRYMSG(e,msg) do{if(!(e)) {LOG("%s(%d): %s"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,msg); goto Error; }}while(0)
#define FAIL     do{          LOG("%s(%d): %s"ENDL "\tExecution should not reach here."ENDL,__FILE__,__LINE__,__FUNCTION__); goto Error; }while(0)
#define TODO     do{          LOG("%s(%d): %s"ENDL "TODO: \tNot implemented yet."ENDL,__FILE__,__LINE__,__FUNCTION__); exit(-1); }while(0)

#ifndef restrict
#define restrict __restrict
#endif
/// @endcond

/** Swaps two pointers */
static void swap(void *a, void *b)
{ void *t=b;b=a;a=t;
}

static int same_except(size_t e,size_t n,const size_t*restrict a,const size_t*restrict b)
{ size_t i;
  for(i=0;i<n;++i) if(i!=e && a[i]!=b[i]) return 0;
  return 1;
}

/** Concatenates arrays.
 *  Appends \a x to \a y along dimension \a idim and returns the result.
 *  Allocates a new array and uses malloc() to allocate the output buffer.
 *  \ingroup ndops
 */
nd_t ndcat(nd_t x, nd_t y, size_t idim)
/// @cond DEFINES
#define LOG(...) do{ ndLogError(x,__VA_ARGS__); ndLogError(y,__VA_ARGS__); } while(0)
/// @endcond
{ nd_t out=0;
  int cleanup=0;
  TRY(ndtype(x)==ndtype(y));// Require input arrays to have the same type
  TRY(ndkind(x)==ndkind(y));// Require input arrays to have the same kind
  REQUIRE(x,PTR_ARITHMETIC|CAN_MALLOC|CAN_REALLOC|CAN_MEMCPY);
  if(ndndim(x)!=ndndim(y))  // do the sensible thing if arrays are different # of dims.
  { if(ndndim(x)>ndndim(y))
      swap(&x,&y);          // ensure x is the array with more dims
    TRY((ndndim(x)-ndndim(y))==1);
    ndInsertDim(y,(unsigned)idim);
    cleanup=1;
  }
  TRY(same_except(idim,ndndim(x),ndshape(x),ndshape(y)));
  ndcast(out=ndinit(),ndtype(x));
  TRY(ndreshape(out,(unsigned)ndndim(x),ndshape(x)));
  ndShapeSet(out,(unsigned)idim,ndshape(out)[idim]+ndshape(y)[idim]);
  TRY(ndref(out,malloc(ndnbytes(out)),nd_heap));
  ndcopy(out,x,0,NULL);
  ndcopy(ndoffset(out,(unsigned)idim,ndshape(x)[idim]),y,0,NULL);
  ndoffset(out,(unsigned)idim,-(int64_t)ndshape(x)[idim]);
Finalize:
  if(cleanup) ndRemoveDim(y,(unsigned)idim);
  return out;
Error:
  ndfree(out);
  out=0;
  goto Finalize;
}
#undef LOG

/** Concatenates arrays "in-place."
 *  Appends \a src to \a dst along the last dimension.  The other dimensions \a
 *  src and \a dst must agree.
 *
 *  This function is more permissive than the out-of-place alternative, ndcat().
 *  It potentially permits arrays of different kinds and types, as dictated by
 *  ndcopy().
 *
 *  Expands the buffer used by \a dst using realloc().
 *  \returns 0 on error, otherwise \a dst.
 *  \ingroup ndops
 */
nd_t ndcat_ip(nd_t dst, nd_t src)
/// @cond DEFINES
#define LOG(...) do{ ndLogError(dst,__VA_ARGS__); } while(0)
/// @endcond
{ nd_t out=0;
  int cleanup=0;
  size_t idim;
  REQUIRE(dst,PTR_ARITHMETIC|CAN_REALLOC|CAN_MEMCPY);
  // handle differing shape.  src is still acceptable as proper subvolume
  if(ndndim(dst)<ndndim(src))
  { TRY((ndndim(src)-ndndim(dst))==1);
    ndInsertDim(dst,(unsigned)ndndim(src)-1);
    cleanup=1;
  } else if(ndndim(dst)>ndndim(src))
  { TRY((ndndim(dst)-ndndim(src))==1);
    ndInsertDim(src,(unsigned)ndndim(dst)-1);
    cleanup=2;
  }
  idim=ndndim(dst)-1;
  TRY(same_except(idim,ndndim(dst),ndshape(src),ndshape(dst)));

  // reshape and realloc
  { size_t o=ndshape(dst)[idim];
    ndShapeSet(dst,(unsigned)idim,ndshape(dst)[idim]+ndshape(src)[idim]);
    TRY(ndref(dst,realloc(nddata(dst),ndnbytes(dst)),nd_heap));
    TRY(ndcopy(ndoffset(dst,(unsigned)idim,o),src,0,NULL));
    out=ndoffset(dst,(unsigned)idim,-(int64_t)o);
  }
Finalize:
  if(cleanup==2)
    ndRemoveDim(src,(unsigned)ndndim(src)-1);
  return out;
Error:
  if(cleanup==1)
    ndRemoveDim(dst,(unsigned)ndndim(dst)-1);
  out=NULL;
  goto Finalize;
}
#undef LOG
