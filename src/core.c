/** \file
    N-Dimensional array type.

    \author Nathan Clack
    \date   June 2012
 */
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
#define FAIL(msg)                    do{ LOG("%s(%d):"ENDL "\t%s"ENDL,__FILE__,__LINE__,msg); goto Error;} while(0)
#define RESIZE(type,e,nelem)         TRY((e)=(type*)realloc((e),sizeof(type)*(nelem)))
#define NEW(type,e,nelem)            TRY((e)=(type*)malloc(sizeof(type)*(nelem)))
#define SAFEFREE(e)                  if(e){free(e); (e)=NULL;}
/// @endcond

/// \brief N-dimensional array type.  Implementation of the abstract type nd_t.
struct _nd_t
{ size_t    ndim;               ///< The number of dimensions
  nd_type_id_t type_desc;       ///< Element type descriptor. \see nd_type_id_t
  nd_kind_t kind;               ///< Kind descriptor. \see nd_kind_t
  size_t   *shape;              ///< Buffer of length ndim,  ordered [w,h,d,...].  Always agrees with stride.  Maintained for convenience.
  size_t   *strides;            ///< Buffer of length ndim+1, strides[i] is the number of bytes layed out between unit steps along dimension i
  char     *log;                ///< If non-null, holds error message log.
  void     *restrict data;      ///< A poitner to the data.
};

/// @cond DEFINES
#undef LOG
#define LOG(...) ndLogError(a,__VA_ARGS__)
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
size_t        ndndim    (const nd_t a)    {return a->ndim;}
const size_t *ndshape   (const nd_t a)    {return a->shape;}
const size_t *ndstrides (const nd_t a)    {return a->strides;}
char*         nderror   (const nd_t a)    {return a->log;}
void          ndResetLog(nd_t a)          {SAFEFREE(a->log);}
nd_kind_t     ndkind    (const nd_t a)    {return a->kind;}

/** \returns the input array \a a*/
nd_t ndsetkind(nd_t a,nd_kind_t kind)
{ a->kind=kind;
  return a;
}

void maybe_resize_array(nd_t a, unsigned ndim)
{ if(a->ndim>=ndim) return; // nothing to do
  RESIZE(size_t,a->shape  ,ndim);
  RESIZE(size_t,a->strides,ndim+1);
  a->ndim=ndim;
Error:
  ;
}

nd_t ndinit(void)
{ nd_t a;
  NEW(struct _nd_t,a,1);
  memset(a,0,sizeof(struct _nd_t));
  maybe_resize_array(a,1);
  a->strides[0]=1;
  return a;
Error:
  return NULL; // if allocation fails, error message is buried.
}

void ndfree(nd_t a)
{ if(!a) return;
  SAFEFREE(a->shape);
  SAFEFREE(a->strides);
  if(a->log) fprintf(stderr,"Log: 0x%p"ENDL "\t%s"ENDL,a,a->log);
  SAFEFREE(a->log);
  free(a);
}

nd_t ndcast(nd_t a, nd_type_id_t desc)
{ size_t o,n,i;
  nd_type_id_t old=a->type_desc;
  maybe_resize_array(a,1);
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

/** initializes \a a so that it references \a buf.

    If the shape of the array \a is already set so that \nelem fits, then
    this just sets the data pointer and returns.

    Otherwise, resizes the array as a 1d container referencing the data.

    \todo Awkward.  I like binding the pointer so that I don't deal with
          memory management inside.  I don't like how this ends up getting
          used.  The ndinit/ndref/ndcast/ndshape pattern sucks.  The
          multiple behaviors (trying to keep shape sometimes, changing it
          other times) also sucks.
*/
nd_t ndref(nd_t a, void *buf, size_t nelem)
{ a->data=buf;
  if(a->strides && ndnelem(a)==nelem)
    return a;
  maybe_resize_array(a,1);
  a->shape[0]=nelem;
  a->strides[0]=ndbpp(a);
  a->strides[1]=ndbpp(a)*nelem;

  return a;
}

/** Reshapes the array.

    The new shape must contain the same number of total elements.
    If the new shape does not conform, an error is generated and
    there is no change to the array.

    \returns 0 on error, otherwise the array \a a.
*/
nd_t ndreshape(nd_t a,unsigned ndim,const size_t *shape)
{ size_t nelem;
  unsigned i;
  for(i=0,nelem=1;i<ndim;++i)
    nelem*=shape[i];
  TRY(nelem==ndnelem(a));

  maybe_resize_array(a,ndim);
  memcpy(a->shape    ,shape,sizeof(*shape)*ndim);
  memcpy(a->strides+1,shape,sizeof(*shape)*ndim);
  a->strides[0]=ndbpp(a);
  for(i=0;i<ndim;++i)
    a->strides[i+1]*=a->strides[i];
  return a;
Error:
  return NULL;
}


/** increments data pointer: data+=o*stride[idim] */
nd_t ndoffset(nd_t a, unsigned idim, int64_t o)
{ TRY(a && a->data);
  TRY(idim<a->ndim);
  a->data=((uint8_t*)a->data)+a->strides[idim]*o;
  return a;
Error:
  return NULL;
}
#pragma warning( pop )
