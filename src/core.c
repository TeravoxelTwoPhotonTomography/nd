/** \file
    N-Dimensional array type and core operations.

    \author Nathan Clack
    \date   June 2012
 */

#include "cuda_runtime_api.h"
#include "nd.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef _MSC_VER
#define va_copy(a,b) (a=b)
#endif

#pragma warning( push )
#pragma warning( disable:4996 ) //unsafe function

/// @cond DEFINES
#define restrict __restrict

#define ENDL                         "\n"
#define LOG(...)                     printf(__VA_ARGS__)
#define TRY(e)                       do{if(!(e)) { LOG("%s(%d): %s"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error;}} while(0)
#define TRYMSG(e,msg)                do{if(!(e)) {LOG("%s(%d): %s"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL "\t%sENDL",__FILE__,__LINE__,__FUNCTION__,#e,msg); goto Error; }}while(0)
#define FAIL(msg)                    do{ LOG("%s(%d):"ENDL "\t%s"ENDL,__FILE__,__LINE__,msg); goto Error;} while(0)
#define RESIZE(type,e,nelem)         TRY((e)=(type*)realloc((e),sizeof(type)*(nelem)))
#define NEW(type,e,nelem)            TRY((e)=(type*)malloc(sizeof(type)*(nelem)))
#define SAFEFREE(e)                  if(e){free(e); (e)=NULL;}
#define CUTRY(e)                     do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,cudaGetErrorString(ecode)); goto Error; }}while(0)
#define CUWARN(e)                    do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,cudaGetErrorString(ecode));             }}while(0)
/// @endcond

#if 0
// signed beg,cur,end should act like they do in python
// (-1) is the end, and so on
struct _nd_slice_t
{ size_t  idim;
  int64_t beg,cur,end,step;
};
#endif

/// \brief N-dimensional array type.  Implementation of the abstract type nd_t.
struct _nd_t
{ size_t    ndim;               ///< The number of dimensions
  size_t   *restrict shape;     ///< Buffer of length ndim,  ordered [w,h,d,...].  Always agrees with stride.  Maintained for convenience.
  size_t   *restrict strides;   ///< Buffer of length ndim+1, strides[i] is the number of bytes layed out between unit steps along dimension i
  void     *restrict data;      ///< A poitner to the data.
  nd_type_id_t type_desc;       ///< Element type descriptor. \see nd_type_id_t
  nd_kind_t kind;               ///< Kind descriptor. \see nd_kind_t
  char     *log;                ///< If non-null, holds error message log.
};

typedef struct _nd_cuda_t* nd_cuda_t;
struct _nd_cuda_t
{ struct _nd_t vol;             ///< host bound shape,strides. device bound data.
  void *restrict dev_shape;   ///< device bound shape array
  void *restrict dev_strides; ///< device bound strides array
};

#include "private/kind.c"

/// @cond DEFINES
#undef LOG
#define LOG(...) ndLogError((nd_t)a,__VA_ARGS__)
/// @endcond

/** Appends message to error log for \a a
    and prints it to \c stderr.
  */
void ndLogError(nd_t a,const char *fmt,...)
{ size_t n,o;
  va_list args,args2;
  if(!a) goto Error;
  va_start(args,fmt);
  va_copy(args2,args);      // [HAX] this is pretty ugly
#ifdef _MSC_VER
  n=_vscprintf(fmt,args)+1; // add one for the null terminator
#else
  n=vsnprintf(NULL,0,fmt,args)+1;
#endif
  va_end(args);
  if(!a->log) a->log=(char*)calloc(n,1);
  //append
  o=strlen(a->log);
  a->log=(char*)realloc(a->log,n+o);
  memset(a->log+o,0,n);
  if(!a->log) goto Error;
  va_copy(args,args2);
  vsnprintf(a->log+o,n,fmt,args2);
  va_end(args2);
#if 0
  vfprintf(stderr,fmt,args);
#endif
  va_end(args);
  return;
Error:
  va_start(args2,fmt);
  vfprintf(stderr,fmt,args2); // print to stderr if logging fails.
  va_end(args2);
  fprintf(stderr,"%s(%d): Logging failed."ENDL,__FILE__,__LINE__);
}

size_t ndbpp(const nd_t a)
{
  static const size_t _Bpp[]={1,2,4,8,1,2,4,8,4,8};
  TRY(a->type_desc>nd_id_unknown);
  TRY(a->type_desc<nd_id_count);
  return _Bpp[(unsigned)a->type_desc];
Error:
  return 0;
}

size_t        ndnelem   (const nd_t a)    {return a->strides[a->ndim]/a->strides[0];}
size_t        ndnbytes  (const nd_t a)    {return a->strides[a->ndim];}
void*         nddata    (const nd_t a)    {return ((uint8_t*)a->data);}
unsigned      ndndim    (const nd_t a)    {return (unsigned)a->ndim;}
size_t*       ndshape   (const nd_t a)    {return a->shape;}
size_t*       ndstrides (const nd_t a)    {return a->strides;}
char*         nderror   (const nd_t a)    {return a?a->log:0;}
void          ndResetLog(nd_t a)          {SAFEFREE(a->log);}
nd_kind_t     ndkind    (const nd_t a)    {return a->kind;}

/** \returns the input array \a a*/
nd_t ndsetkind(nd_t a,nd_kind_t kind)
{ a->kind=kind;
  return a;
}

/** Resizes strides and shapes array, but does not initialize memory. */
static
void maybe_resize_array(nd_t a, unsigned ndim)
{ size_t odim=a->ndim;
  if(a->ndim>=ndim) { a->ndim=ndim; return;} // nothing to do
  RESIZE(size_t,a->shape  ,ndim);
  RESIZE(size_t,a->strides,ndim+1);
  a->ndim=ndim;
Error:
  ;
}

/** Produces an empty array.
 *  An empty array has zero dimension.
 */
nd_t ndinit(void)
{ nd_t a;
  NEW(struct _nd_t,a,1);
  memset(a,0,sizeof(struct _nd_t));
  NEW(size_t,a->strides,1);
  a->strides[0]=1;
  return a;
Error:
  return NULL; // if allocation fails, error message is buried.
}

/**
 * This is called from ndfree() before ndfree() does anything.
 */
static void ndcuda_free(nd_cuda_t a)
{ void *d=0;
  if(d=nddata((nd_t)a))CUWARN(cudaFree(d));
  if(d=a->dev_shape)   CUWARN(cudaFree(d));
  if(d=a->dev_strides) CUWARN(cudaFree(d));
}

void ndfree(nd_t a)
{ if(!a) return;
  switch(ndkind(a)) // specially handle certain kinds
  { case nd_gpu_cuda: ndcuda_free((nd_cuda_t)a); break;
    //case nd_heap:     if(a->data) free(a->data); break;
    default:;
  }
  SAFEFREE(a->shape);
  SAFEFREE(a->strides);
  if(a->log) fprintf(stdout,"Log: 0x%p"ENDL "\t%s"ENDL,a,a->log);
  SAFEFREE(a->log);
  free(a);
}

nd_t ndcast(nd_t a, nd_type_id_t desc)
{ size_t o,n,i;
  nd_type_id_t old=a->type_desc;
  //maybe_resize_array(a,1);
  TRY(o=ndbpp(a));
  a->type_desc = desc;
  TRY(n=ndbpp(a)); // checks for valid descriptor
  for(i=0;i<a->ndim+1;++i) a->strides[i]/=o; // adjust strides
  for(i=0;i<a->ndim+1;++i) a->strides[i]*=n;
  return a;
Error:
  a->type_desc = old;
  return 0;
}

nd_type_id_t ndtype(const nd_t a)
{ return a->type_desc;
}

/** Initializes \a a so that it references \a buf.

    If buf is NULL, the data pointer is set to NULL, but an error is 
    logged and the function returns NULL.

    This supports clearing the data pointer like this:

    \code{C}
    ndref(a,0,0);
    \encode

    but permits catching errors like this:
    
    \code{C}
    if(!ndref(a,malloc(1024),1024)) goto Error;
    \endcode

    If the shape of the array \a is already set so that \nelem fits, then
    this just sets the data pointer and returns.

    Otherwise, resizes the array as a 1d container referencing the data.

    \todo Only take's nelem as an argument in case a is not initialized.
          I think I should eliminate the nelem argument, leaving the
          size as 1.

    \todo Awkward.  I like binding the pointer so that I don't deal with
          memory management inside.  I don't like how this ends up getting
          used.  The ndinit/ndref/ndcast/ndshape pattern sucks.  The
          multiple behaviors (trying to keep shape sometimes, changing it
          other times) also sucks.
*/
nd_t ndref(nd_t a, void *buf, size_t nelem)
{ TRY(a->data=buf);
  if(a->strides && ndnelem(a)==nelem)
    return a;
  if(a->ndim==0)
  { maybe_resize_array(a,1);
    a->shape[0]=nelem;
    a->strides[0]=ndbpp(a);
    a->strides[1]=ndbpp(a)*nelem;
  }
  return a;
Error:
  return NULL;
}

/** Reshapes the array.
 *
 *  This function assumes you know what you're doing with the shape.  It does
 *  not do any bounds checking.
 *
 *  Can use this to change the number of dimensions by calling
 *  \code{c}
 *  ndreshape(array,ndim,ndndim(array))
 *  \endcode
 *
 *  \returns 0 on error, otherwise the array \a a.
 */
nd_t ndreshape(nd_t a,unsigned ndim,const size_t *shape)
{ unsigned i;
  //size_t nelem;
  //for(i=0,nelem=1;i<ndim;++i)
  // nelem*=shape[i];
  //TRY(nelem<=ndnelem(a));
  maybe_resize_array(a,ndim); // sets ndim, maybe resizes shape/strides 
  if(a->shape==shape) return a;
  memcpy(a->shape    ,shape,sizeof(*shape)*ndim);
  memcpy(a->strides+1,shape,sizeof(*shape)*ndim);
  a->strides[0]=ndbpp(a);
  for(i=0;i<ndim;++i)
    a->strides[i+1]*=a->strides[i];
  return a;
//Error:
//  return NULL;
}

/** Sets the shape for dimension \a idim to \a val.
 *  Updates the strides to reflects the new shape.
 *
 *  The arrays dimensionality will be increased to fit \a idim if necessary.
 *
 *  \param[in]    a     The array on which to operate.
 *  \param[in]    idim  The dimension to change.
 *  \param[in]    val   The new size of the dimension \a idim.
 *  \returns The input array \a a.
 */
nd_t ndShapeSet(nd_t a, unsigned idim, size_t val)
{ size_t i;
  if(idim>=a->ndim)
    TRY(ndInsertDim(a,idim));
  a->shape[idim]=val;
  memcpy(a->strides+1,a->shape,sizeof(*a->shape)*a->ndim);
  a->strides[0]=ndbpp(a);
  for(i=0;i<a->ndim;++i)
    a->strides[i+1]*=a->strides[i];
  return a;
Error:
  return NULL;
}

/// @cond DEFINES
#undef LOG
#define LOG(...) ndLogError(a,__VA_ARGS__)
/// @endcond

/** Inserts an extra dimension at \a idim.
 *  The new dimension will have size 1.
 *
 *  Example:
 *  \code{c}
 *  ndInsertDim(a,x);
 *  ndshape(a)[x]==1; // should be true
 *  \endcode
 */
nd_t ndInsertDim(nd_t a, unsigned idim)
{ size_t i,
         odim=a->ndim,
         ndim=((idim<odim)?odim:idim)+1;
  maybe_resize_array(a,(unsigned)ndim);
  // pad out with singleton dims if necessary
  for(i=odim+1;i<=ndim;++i) a->strides[i]=a->strides[odim];
  for(i=odim;i<ndim;++i) a->shape[i]=1;   // shape[x]=stride[x+1]/stride[x]
  // insert singleton dimension at idim
  memmove(a->shape+idim+1,a->shape+idim,sizeof(*(a->shape))*(ndim-idim-1));
  a->shape[idim]=1;
  memmove(a->strides+idim+1,a->strides+idim,sizeof(*(a->strides))*(ndim-idim));
  return a;
}

/** Removes dimension \a idim merging it with the next dimension.
 */
nd_t ndRemoveDim(nd_t a, unsigned idim)
{ size_t odim;
  if((odim=ndndim(a))<=idim) return a; // do nothing
  if(odim!=idim+1)
  { a->shape[idim+1]*=a->shape[idim];
    memmove(a->shape+idim,a->shape+idim+1,sizeof(*(a->shape))*(odim-idim-1));
    memmove(a->strides+idim+1,a->strides+idim+2,sizeof(*(a->strides))*(odim-idim-1));
  }
  --a->ndim;
  return a;
}

/** increments data pointer: data+=o*stride[idim] */
nd_t ndoffset(nd_t a, unsigned idim, int64_t o)
{ TRY(a && a->data);
  TRY(idim<a->ndim);
  REQUIRE(a,PTR_ARITHMETIC);
  a->data=((uint8_t*)a->data)+a->strides[idim]*o;
  return a;
Error:
  return NULL;
}

//
// === CUDA ===
//

/// @cond DEFINES
#undef LOG
#define LOG(...) fprintf(stderr,__VA_ARGS__)
/// @endcond

static nd_cuda_t ndcuda_init(void)
{ nd_cuda_t out=0;
  nd_t tmp=0;
  NEW(struct _nd_cuda_t,out,1);
  memset(out,0,sizeof(struct _nd_cuda_t));
  TRY(tmp=ndinit());
  memcpy(out,tmp,sizeof(*tmp));
  free(tmp);
  TRY(ndsetkind((nd_t)out,nd_gpu_cuda));
  return out;
Error:
  return 0;
}

/// @cond DEFINES
#undef LOG
#define LOG(...) ndLogError((nd_t)a,__VA_ARGS__)
/// @endcond

/**
 * Allocate a gpu based array according to the shape specified by \a a.
 * Does *not* upload data.  Only uploads shape and strides.
 * The caller is responsible for deallocating the returned array using ndfree().
 * If stream is not NULL, will use the async api for copying memeory to the GPU.
 */
nd_t ndcuda(nd_t a,cudaStream_t stream)
{ nd_cuda_t out;
  TRY(out=ndcuda_init());
  TRY(ndreshape(ndcast((nd_t)out,ndtype(a)),(unsigned)ndndim(a),ndshape(a)));
  
  CUTRY(cudaMalloc((void**)&out->dev_shape  ,sizeof(size_t)* ndndim(a)   )); // the extra void** cast is to satisfy compilers that complain about discarding __restrict__
  CUTRY(cudaMalloc((void**)&out->dev_strides,sizeof(size_t)*(ndndim(a)+1)));
  CUTRY(cudaMalloc((void**)&out->vol.data   ,ndnbytes(a)));

  TRY(ndCudaSyncShape((nd_t)out,stream));
  return (nd_t)out;
Error:  
  if(out) free(out);  // I suppose ndfree should know how to free gpu-based shape and strides
  return 0;
}

/**
 * Copies host-based shapes and strides to the GPU.
 * \todo bad name: can't tell from name direction of transfer
 */
nd_t ndCudaSyncShape(nd_t a,cudaStream_t stream)
{ nd_cuda_t self=(nd_cuda_t)a;
  REQUIRE(a,CAN_CUDA);
  CUTRY(cudaMemcpy(self->dev_shape  ,a->shape  ,sizeof(size_t)* ndndim(a)   ,cudaMemcpyHostToDevice));
  CUTRY(cudaMemcpy(self->dev_strides,a->strides,sizeof(size_t)*(ndndim(a)+1),cudaMemcpyHostToDevice));
  return a;
Error:
  return 0;
}

/// @cond DEFINES
#undef LOG
#define LOG(...) ndLogError((nd_t)dst,__VA_ARGS__)
/// @endcond

/**
 * Copies data to/from the GPU from/to a host array.
 * 
 * Stream may be 0.  Specifies the stream to use for async copies.
 * Async copies are the default.
 */
nd_t ndCudaCopy(nd_t dst, nd_t src,cudaStream_t stream)
{ enum cudaMemcpyKind direction;
  size_t sz;
  if(ndkind(dst)==nd_gpu_cuda)
    { REQUIRE(src,CAN_MEMCPY);
      direction=cudaMemcpyHostToDevice;
      sz=ndnbytes(src);
    }
  else if(ndkind(src)==nd_gpu_cuda)
    { REQUIRE(dst,CAN_MEMCPY);
      direction=cudaMemcpyDeviceToHost;
      sz=ndnbytes(dst);
      //CUTRY(cudaStreamSynchronize(stream)); // sync before read
    }
  else
    FAIL("Only Host-to-Device or Device-to-Host copies are supported");
  //CUTRY(cudaMemcpyAsync(nddata(dst),nddata(src),sz,direction,stream));
  CUTRY(cudaMemcpy(nddata(dst),nddata(src),sz,direction));
  return dst;
Error:
  return 0;
}

/**
 * GPU based shape array.349
 * 
 * \returns device pointer on success, otherwise 0.
 */
void* ndCudaShape    (nd_t self) {return (self&&(ndkind(self)==nd_gpu_cuda))?((nd_cuda_t)self)->dev_shape  :0;}

/**
 * GPU based strides array.
 * \returns device pointer on success, otherwise 0.
 */
void* ndCudaStrides  (nd_t self) {return (self&&(ndkind(self)==nd_gpu_cuda))?((nd_cuda_t)self)->dev_strides:0;}

#pragma warning( pop )
