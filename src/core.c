/** \file
    N-Dimensional array type and core operations.

    \author Nathan Clack
    \date   June 2012
 */
#include "config.h"
#if HAVE_CUDA
  #include "cuda_runtime_api.h"
#endif
#include "nd.h"
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#ifdef _MSC_VER
#include <malloc.h>
#define alloca _alloca
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
#define ZERO(type,e,nelem)           memset((e),0,sizeof(type)*(nelem))
#define SAFEFREE(e)                  if((e)){free(e); (e)=NULL;}
#if HAVE_CUDA
  #define CUTRY(e)                     do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,cudaGetErrorString(ecode)); goto Error; }}while(0)
  #define CUWARN(e)                    do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,cudaGetErrorString(ecode));             }}while(0)
#else
  #define CUTRY(e)  TRY(e)
  #define CUWARN(e) TRY(e)
#endif
/// @endcond

#if 0
// signed beg,cur,end should act like they do in python
// (-1) is the end, and so on
struct _nd_slice_t
{ size_t  idim;
  int64_t beg,cur,end,step;
};
#endif

struct _nd_t;

typedef struct _nd_stack_t
{ int           i,
                sz;
  struct _nd_t* data; 
} nd_stack_t;

/// \brief N-dimensional array type.  Implementation of the abstract type nd_t.
struct _nd_t
{ size_t       ndim;               ///< The number of dimensions
  size_t      *restrict shape;     ///< Buffer of length ndim,  ordered [w,h,d,...].  Always agrees with stride.  Maintained for convenience.
  size_t      *restrict strides;   ///< Buffer of length ndim+1, strides[i] is the number of bytes layed out between unit steps along dimension i
  void        *restrict data;      ///< A poitner to the data.
  nd_type_id_t type_desc;          ///< Element type descriptor. \see nd_type_id_t
  nd_kind_t    kind;               ///< Kind descriptor. \see nd_kind_t
  char        *log;                ///< If non-null, holds error message log.
  nd_stack_t   history;
};

typedef struct _nd_cuda_t* nd_cuda_t;
/// Castable to nd_t.
struct _nd_cuda_t
{ struct _nd_t vol;             ///< host bound shape,strides. device bound data.  Must be the first element in the struct.
#if HAVE_CUDA                                
  cudaStream_t stream;          ///< currently bound stream
#else
  int          stream;
#endif
  size_t         dev_ndim;      ///< dimension of device-bound shape
  size_t         dev_cap;       ///< capacity of device buffer (aids reallocation)
  void *restrict dev_shape;     ///< device bound shape array
  void *restrict dev_strides;   ///< device bound strides array
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

size_t        ndnelem   (const nd_t a)    {return a?a->strides[a->ndim]/a->strides[0]:0;}
size_t        ndnbytes  (const nd_t a)    {return a?a->strides[a->ndim]:0;}
void*         nddata    (const nd_t a)    {return a?((uint8_t*)a->data):0;}
unsigned      ndndim    (const nd_t a)    {return a?(unsigned)a->ndim:0;}
size_t*       ndshape   (const nd_t a)    {return a?a->shape:0;}
size_t*       ndstrides (const nd_t a)    {return a?a->strides:0;}
char*         nderror   (const nd_t a)    {return a?a->log:0;}
void          ndResetLog(nd_t a)          {SAFEFREE(a->log);}
nd_kind_t     ndkind    (const nd_t a)    {return a?a->kind:nd_unknown_kind;}

/*
 * SHAPE HISTORY 
 */

/** \returns a on success, 0 on error */
static nd_t push(nd_stack_t *stack, nd_t a)
{ nd_t c=0;
  if(stack->i>=stack->sz)
  { stack->sz=stack->i*1.2+50;
    RESIZE(struct _nd_t,stack->data,stack->sz);
  }
  ZERO(struct _nd_t,(c=stack->data+stack->i),1);
  stack->i++; // i points to the next write point, 1 past the read point
  // Make a partial copy of a.  Want the shape, strides, offset, and dimensionality.
  c->ndim=a->ndim;
  NEW(size_t,c->shape,c->ndim);
  NEW(size_t,c->strides,c->ndim+1);
  memcpy(c->shape,a->shape,c->ndim*sizeof(size_t));
  memcpy(c->strides,a->strides,(c->ndim+1)*sizeof(size_t));
  c->data=a->data;
  c->kind=nd_unknown_kind;
  return a;
Error:
  return 0;
}

/** \returns a on success, or 0 on underflow. */
static nd_t pop (nd_stack_t *stack, nd_t a)
{ nd_t c=0;
  TRY(stack->sz==0 || stack->i<stack->sz);
  if(stack->i==0) return 0; //underflow
  c=stack->data+(--stack->i);
  a->ndim=c->ndim;
  memcpy(a->shape  ,c->shape,c->ndim*sizeof(size_t));
  memcpy(a->strides,c->strides,(c->ndim+1)*sizeof(size_t));
  free(c->shape);
  free(c->strides);
  a->data=c->data;
  return a;
Error:
  return 0;
}

/** Saves the shape to the a's internal shape stack.
    \returns a on success, 0 otherwise.
*/
nd_t ndPushShape(nd_t a)
{ TRY(a);
  TRY(push(&a->history,a));
  return a;
Error:
  return 0;
}

nd_t ndPopShape (nd_t a)
{ TRY(a);
  TRY(pop(&a->history,a));
#if 0 // not sure if this is a good idea or not...I think shape sync should be explicit
  if(a->kind==nd_gpu_cuda)
    TRY(ndCudaSyncShape(a));
#endif
  return a;
Error:
  return 0;
}

/** Fills buf with the shape cast to integers.
    \param  buf must be at least ndndim(a) elements in size.
    \return buf on success, otherwise 0. 
*/
int* ndshape_as_int(nd_t a, int *buf)
{ unsigned i;
  size_t *s=ndshape(a);
  if(!s) return 0;
  for(i=0;i<ndndim(a);++i) buf[i]=(int)s[i];
  return buf;
}

/** Fills buf with the strides cast to integers.
    \param  buf must be at least ndndim(a)+1 elements in size.
    \return buf on success, otherwise 0. 
*/
int* ndstrides_as_int(nd_t a, int *buf)
{ unsigned i;
  size_t *s=ndstrides(a);
  if(!s) return 0;
  for(i=0;i<ndndim(a)+1;++i) buf[i]=(int)s[i];
  return buf;
}

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
  a->kind=nd_unknown_kind;
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
#if HAVE_CUDA
  if((d=nddata((nd_t)a)))CUWARN(cudaFree(d));
  if((d=a->dev_shape))   CUWARN(cudaFree(d));
  if((d=a->dev_strides)) CUWARN(cudaFree(d));
#endif
}

void ndfree(nd_t a)
{ if(!a) return;
  while(pop(&a->history,a)); // not necessary to unroll, could just jump to original
  if(a->history.data) free(a->history.data);
  switch(ndkind(a)) // specially handle certain kinds
  { case nd_gpu_cuda: ndcuda_free((nd_cuda_t)a); break;
    case nd_heap:     if(a->data) free(a->data); break;
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
{ return a?a->type_desc:nd_id_unknown;
}

/** Initializes \a a so that it references \a buf.

    If buf is NULL, the data pointer is set to NULL, but an error is 
    logged and the function returns NULL.

    This supports clearing the data pointer like this:

    \code{C}
    ndref(a,0,nd_unknown_kind);
    \encode

    but permits catching errors like this:
    
    \code{C}
    if(!ndref(a,malloc(1024),nd_heap)) goto Error;
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
nd_t ndref(nd_t a, void *buf, nd_kind_t kind)
{ TRY(a);
  a->data=buf;
  a->kind=kind;
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
  maybe_resize_array(a,ndim); // sets ndim, maybe resizes shape/strides 
  if(a->shape!=shape)
    memcpy(a->shape,shape,sizeof(*shape)*ndim);
  //update strides
  memcpy(a->strides+1,shape,sizeof(*shape)*ndim);
  a->strides[0]=ndbpp(a);
  for(i=0;i<ndim;++i)
    a->strides[i+1]*=a->strides[i];
  return a;
}

nd_t ndreshapev(nd_t a,unsigned ndim,...)
{ size_t *shape=0;
  unsigned i;
  va_list args;  
  TRY(shape=(size_t*)alloca(ndim*sizeof(size_t)));
  va_start(args,ndim);
  for(i=0;i<ndim;++i) shape[i]=va_arg(args,unsigned);
  va_end(args);
  return ndreshape(a,ndim,shape);
Error:
  return 0;
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

/**
 * Swaps the contents of two arrays.  Simply copies the contents of the 
 * nd_t struct back and forth.
 */
void ndswap(nd_t a, nd_t b)
{ if(a&&b)
  { struct _nd_t t=*a;
    *a=*b;
    *b=t;
  }
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
#if HAVE_CUDA
  NEW(struct _nd_cuda_t,out,1);
  memset(out,0,sizeof(struct _nd_cuda_t));
  TRY(tmp=ndinit());
  memcpy(out,tmp,sizeof(*tmp));
  free(tmp);
  TRY(ndsetkind((nd_t)out,nd_gpu_cuda));
  return out;
#else
  FAIL("[libnd] CUDA support unavaible."); 
#endif
Error:
  return 0;
}

/// @cond DEFINES
#undef LOG
#define LOG(...) ndLogError((nd_t)a,__VA_ARGS__)
/// @endcond

/**
 * Constructor.  Creates a new array and allocates space according to the type 
 * and shape specified by \a a.
 *
 * Note that not all kinds are supported.
 * 
 * Note that data isn't copied.  To allocate and generate a copy do:
 * \code{c}
 * nd_t b=ndcopy(ndmake_kind(a,nd_heap),a,0,0);
 * \endcode
 */
nd_t ndmake_kind(nd_t a,nd_kind_t kind)
{ switch(kind)
  { case nd_id_unknown:return ndunknown(a);
    case nd_heap:      return ndheap(a);
    case nd_gpu_cuda:  return ndcuda(a,0);
    default:
      FAIL("No constructor found for the requested array kind.");
  }
Error:
  return 0;
}

/**
 * Constructor.  Creates a new array and allocates space according to the kind 
 * and shape specified by \a a.
 *
 * Note that not all kinds are supported.
 * 
 * Data isn't copied.  To allocate and generate a copy do:
 * \code{c}
 * nd_t b=ndcopy(ndmake_type(a,nd_u8),a,0,0);
 * \endcode
 */
nd_t ndmake_type(nd_t a,nd_type_id_t type)
{ nd_t out=0;
  nd_type_id_t old=ndtype(a);
  TRY(a=ndcast(a,type));
  switch(ndkind(a))
  { case nd_id_unknown: out=ndunknown(a); break;
    case nd_heap:       out=ndheap(a); break;
    case nd_gpu_cuda:   out=ndcuda(a,0); break;
    default:
      FAIL("No constructor found for the requested array kind.");
  }
  ndcast(a,old);
  return out;
Error:
  return 0;
}

/**
 * Constructor.  Creates a new array and allocates space according to the kind 
 * and shape specified by \a a.
 *
 * Note that not all kinds are supported.
 * 
 * Note that data isn't copied.  To allocate and generate a copy do:
 * \code{c}
 * nd_t b=ndcopy(ndheap(a),a,0,0);
 * \endcode
 */
nd_t ndmake(nd_t a) {return ndmake_kind(a,ndkind(a));}

/**
 * Constructor.  Creates an array according to the shape specified by \a a.
 * Does not allocate any data.
 *
 * This is useful for setting the shape and type, and defering allocation till 
 * later.
 */
nd_t ndunknown(nd_t a)
{ nd_t out;
  TRY(ndreshape(ndcast(out=ndinit(),ndtype(a)),ndndim(a),ndshape(a)));
  TRY(ndref(out,0,nd_unknown_kind));
  return out;
Error:
  return 0;
}

/**
 * Constructor.  Creates a RAM based array according to the shape specified by \a a.
 * Allocates the data buffer with malloc();
 * Note that data isn't copied.  To allocate and genrate a copy do:
 * \code{c}
 * nd_t b=ndcopy(ndheap(a),a,0,0);
 * \endcode
 */
nd_t ndheap(nd_t a)
{ nd_t out;
  TRY(a);
  TRY(ndreshape(ndcast(out=ndinit(),ndtype(a)),ndndim(a),ndshape(a)));
  TRY(ndref(out,malloc(ndnbytes(out)),nd_heap));
  return out;
Error:
  return 0;
}

/**
 * Allocator.  Creates a RAM based array according to the shape specified by \a a.
 * Changes the \a a data pointer.
 * Allocates the data buffer with realloc();
 * Example:
 * \code{c}
 * nd_t a=0;
 * a=ndheap_ip(ndioShape(f));
 * \endcode
 */
nd_t ndheap_ip (nd_t a)
{ TRY(a);
  TRY(ndreshape(a,ndndim(a),ndshape(a)));
  TRY(ndref(a,realloc(nddata(a),ndnbytes(a)),nd_heap));
  return a;
Error:
  return 0;

}

/**
 * Allocate a gpu based array according to the shape specified by \a a.
 * Does *not* upload data.  Only uploads shape and strides.
 * The caller is responsible for deallocating the returned array using ndfree().
 * If stream is not NULL, will use the async api for copying memeory to the GPU.
 */
nd_t ndcuda(nd_t a,void* stream)
{ 
#if HAVE_CUDA
  nd_cuda_t out=0;
  TRY(out=ndcuda_init());  
  out->stream=(cudaStream_t)stream;
  TRY(ndreshape(ndcast((nd_t)out,ndtype(a)),(unsigned)ndndim(a),ndshape(a)));
  
  CUTRY(cudaMalloc((void**)&out->dev_shape  ,sizeof(size_t)* ndndim(out)   ));
  CUTRY(cudaMalloc((void**)&out->dev_strides,sizeof(size_t)*(ndndim(out)+1)));
  CUTRY(cudaMalloc((void**)&out->vol.data   ,out->dev_cap=ndnbytes(out)));
  out->dev_ndim=ndndim(out);

  TRY(ndCudaSyncShape((nd_t)out));
#else // fall back to heap
  nd_t out=0;
  TRY(out=ndheap(a));
  ndLogError(out,"Warning: CUDA storage requested but not available.  Defaulting to CPU."ENDL);
#endif
  return (nd_t)out;
Error:  
  if(out) free(out);  // I suppose ndfree should know how to free gpu-based shape and strides
  return 0;
}

/** memset() for the gpu-based nd_t array.
 *  A thin wrapper around CUDA's cudaMemset function.
 *  \returns the input array, \a a, on sucess.  Otherwise 0.
 */
nd_t ndCudaMemset(nd_t a, unsigned char v)
{ 
  REQUIRE(a,CAN_CUDA);
#if HAVE_CUDA
  CUTRY(cudaMemset(nddata(a),v,ndnbytes(a)));
  return a;
#else
  FAIL("[libnd] CUDA support unavailable");
#endif
Error:
  return 0;
}

/**
 * Copies host-based shapes and strides to the GPU.
 * \todo bad name: can't tell from name direction of transfer
 */
nd_t ndCudaSyncShape(nd_t a)
{ nd_cuda_t self=(nd_cuda_t)a;
  REQUIRE(a,CAN_CUDA);
#if HAVE_CUDA
  if(self->dev_ndim<ndndim(a)) // resize device shape and strides arrays if necessary
  { CUTRY(cudaFree(self->dev_shape));
    CUTRY(cudaFree(self->dev_strides));
    CUTRY(cudaMalloc((void**)&self->dev_shape  ,sizeof(size_t)* ndndim(a)   ));
    CUTRY(cudaMalloc((void**)&self->dev_strides,sizeof(size_t)*(ndndim(a)+1)));
    self->dev_ndim=ndndim(a);
  }
  CUTRY(cudaMemcpy(self->dev_shape  ,a->shape  ,sizeof(size_t)* ndndim(a)   ,cudaMemcpyHostToDevice));
  CUTRY(cudaMemcpy(self->dev_strides,a->strides,sizeof(size_t)*(ndndim(a)+1),cudaMemcpyHostToDevice));
  TRY(ndCudaSetCapacity(a,ndnbytes(a)));
  return a;
#else
  FAIL("[libnd] CUDA support unavailable");
#endif
Error:
  return 0;
}

/// @cond DEFINES
#undef LOG
#define LOG(...) ndLogError((nd_t)dst,__VA_ARGS__)
/// @endcond

/**
 * GPU based shape array.
 * \returns device pointer on success, otherwise 0.
 */
void* ndCudaShape    (nd_t self) {return (self&&(ndkind(self)==nd_gpu_cuda))?((nd_cuda_t)self)->dev_shape  :0;}

/**
 * GPU based strides array.
 * \returns device pointer on success, otherwise 0.
 */
void* ndCudaStrides  (nd_t self) {return (self&&(ndkind(self)==nd_gpu_cuda))?((nd_cuda_t)self)->dev_strides:0;}

#undef LOG
#define LOG(...) ndLogError(self_,__VA_ARGS__)

void* ndCudaStream(nd_t self_)
{ return (void*)(((nd_cuda_t)self_)->stream);
}

/**
 * Set the stream to be used for asynchronous operations on the array.
 * \param[in]   self    The nd_t array.
 * \param[in]   stream  May be 0. A Cuda stream identifier.
 * \returns \a self (always succeeeds).
 */
nd_t ndCudaBindStream(nd_t self_, void* stream)
{ nd_cuda_t self=(nd_cuda_t)self_;
#if HAVE_CUDA
  self->stream=(cudaStream_t)stream;
#else
  FAIL("[libnd] CUDA support unavailable");
#endif
  return self_;
Error:
  return 0;
}

/** Synchronize on the currently bound stream. */
nd_t ndCudaWait(nd_t self_)
{ nd_cuda_t self=(nd_cuda_t)self_;
#if HAVE_CUDA
  if(self->stream)
    CUTRY(cudaStreamSynchronize(self->stream));
#endif
  return self_;
Error:
  return 0;
}

/**
 * Free's the old gpu-based buffer referenced by \a self, and allocates a new
 * one with size \a nbytes.
 *
 * Does nothing else; the new allocation size is not reflected in the shape or
 * strides of the array.
 *
 * Also, in contrast to the traditional realloc(), this does NOT copy the old 
 * data to the new buffer.
 */
nd_t ndCudaSetCapacity(nd_t self_, size_t nbytes)
{ nd_cuda_t self=(nd_cuda_t)self_;
#if HAVE_CUDA
  if(self->dev_cap<nbytes)
  { CUTRY(cudaFree(self_->data));
    CUTRY(cudaMalloc((void**)&self_->data,nbytes));
    self->dev_cap=nbytes;
  }
#endif
  return self_;
Error:
  return 0;
}

#pragma warning( pop )
