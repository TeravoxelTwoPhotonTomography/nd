/** \file
    nD Array IO facilities.

    Currently never unloads the shared libaries loaded by the plugin system.

    \todo change api so read and write accept shape arguments.
    \todo check for NULL's in a plugin's api before calling.  Some functions
          may not be implemented.

    \author Nathan Clack
    \date   June 2012
*/
#pragma warning( push )
#pragma warning( disable:4996 ) //unsafe function, use non-standard secure version

#include "nd.h"
#include "io.h"
#include "io/interface.h"
#include "io/plugin.h"
#include "config.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#ifdef _MSC_VER
#define va_copy(a,b) (a=b)
#define alloca _alloca
#endif

/// @cond DEFINES
#define ENDL                         "\n"
#define LOG(...)                     fprintf(stderr,__VA_ARGS__)
#define HERE                         LOG("HERE -- %s(%d): %s()"ENDL,__FILE__,__LINE__,__FUNCTION__)
#define TRY(e)                       do{if(!(e)) { LOG("%s(%d): %s()"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error;}} while(0)
#define NEW(type,e,nelem)            TRY((e)=(type*)malloc(sizeof(type)*(nelem)))
#define SAFEFREE(e)                  if(e){free(e); (e)=NULL;}
/// @endcond

static ndio_fmts_t g_formats=NULL;      ///< List of loaded plugins
static size_t      g_countof_formats=0; ///< The number of loaded plugins

/** File handle.
 *  Implementation for the abstract type, \a ndio_t.
 */
struct _ndio_t
{ ndio_fmt_t *fmt;       ///< The plugin API used to operate on the file.
  void       *context;   ///< The data that the plugin uses to operate on the file.  A plugin-specific file handle.

  nd_t        shape;     ///< (subarrays) 
  nd_t        cache;     ///< (subarrays) Cache used to store temporary data.
  char       *seekable;  ///< (subarrays) Seekable dimension mask
  size_t      seekable_n;///< (subarrays) count of elements allocated for \a seekable
  size_t     *dstpos;    ///< (subarrays) Destination position index
  size_t      dstpos_n;  ///< (subarrays) count of elements allocated for \a dstpos
  size_t     *srcpos;    ///< (subarrays) File positions index
  size_t      srcpos_n;  ///< (subarrays) count of elements allocated for \a srcpos
  size_t     *cachepos;  ///< (subarrays) Cache origin in file space
  size_t      cachepos_n;///< (subarrays) count of elements allocated for \a cachepos

  char       *log;       ///< Used to store the error log.  NULL if no errors, otherwise a NULL terminated string.
};

//
// === HELPERS ===
//

/** \todo make thread safe, needs a mutex */
static int maybe_load_plugins()
{ if(!g_formats)
  { TRY(ndioAddPluginPath(NDIO_PLUGIN_PATH));
    TRY(g_formats=ndioLoadPlugins(NULL,&g_countof_formats));
  }
  return 1;
Error:
  return 0;
}

/** \returns the index of the detected format on sucess, otherwise -1 */
static int detect_file_type(const char *filename, const char *mode)
{ size_t i;
  TRY(filename);
  TRY(mode);
  for(i=0;i<g_countof_formats;++i)
    if(g_formats[i]->is_fmt(filename,mode))
      return (int)i;
Error:
  return -1;
}

/** \returns the index of the detected format on sucess, otherwise -1 */
static int get_format_by_name(const char *format)
{ size_t i;
  if(!format) return -1;
  for(i=0;i<g_countof_formats;++i)
    if(0==strcmp(format,g_formats[i]->name()))
      return (int)i;
  return -1;
}

//
// === INTERFACE ===
//

ndio_fmt_t** ndioPlugins(size_t *count)
{ if(count) *count=g_countof_formats;
  return g_formats;
}

void* ndioContext(ndio_t file) { return file?file->context:0; }
char* ndioError  (ndio_t file) { return file?file->log:0; }

/** Preload's plugins from the default plugin path.
 *
 *  If the plugins have already been loaded, this does nothing.
 *
 *  Normally, calling this function is not required; plugin's are loaded
 *  automatically on demand.
 *
 *  However, when a plugin is loaded, it may load another library that requires
 *  it's initialization is done only in the master thread.  For example, this
 *  is true of the CoreVideo framework on OS X, which is used by FFMPEG.
 *
 *  \returns 1 on success, 0 otherwise.
 */
int ndioPreloadPlugins()
{ return maybe_load_plugins();
}

/** Adds the interface specified by \a plugin to the internal list of file
 *  format interfaces.
 *
 *  This is useful for adding a custom reader at run time, for example, for 
 *  loading files from a specific directory structure or specialized database.
 *
 *  If the plugin->is_fmt() function always returns false, \a plugin will not
 *  interfere with automatic format detection.  It can be addressed
 *  specifically, by using ndioOpen() with the plugin's name (as returned by
 *  plugin->name()).
 *
 *
 * \param[in] plugin  Should be allocated using malloc().  The ndio library
 *                    takes ownership of the object.  Be sure \c plugin->lib is
 *                    set to 0 if the interface did not come from a shared
 *                    library load.
 */
unsigned ndioAddPlugin(ndio_fmt_t* plugin)
{ TRY(g_formats=realloc(g_formats,sizeof(*g_formats)*(g_countof_formats+1)));
  g_formats[g_countof_formats++]=plugin;
  return 1;
Error:
  return 0;
}

/** Determines if the file can be read by any of the file formats. */
int ndioIsFile(const char *filename)
{ maybe_load_plugins();
  return detect_file_type(filename,"r")>=0;
}


/** Opens a file according to the mode.

    \param[in] filename The path to the file.
    \param[in] format   The name of the desired format, or NULL.  If NULL,
                        an attempt will be made to infer the file format
                        from the file itself (if present) or the filename.
                        If no matching format is found, a default format
                        will be used to write the file(mode w), but a read
                        (mode r) will fail.
    \param[in] mode     May be "r" to open the file for reading or "w" to
                        open the file for writing.

    \returns NULL on failure, otherwise an ndio_t object.
 */
ndio_t ndioOpen(const char* filename, const char *format, const char *mode)
{ ndio_t file=NULL;
  void *ctx=NULL;
  int ifmt;
  maybe_load_plugins();
  // Some input validation.  This relieves plugins from having to handle these cases.
  TRY(filename);
  TRY(*filename); //assert non-empty string
  TRY(mode);
  TRY(*mode);     //assert non-empty string
  if(format)
  { if(0>(ifmt=get_format_by_name(format))) goto ErrorSpecificFormat;
  } else
  { if(0>(ifmt=detect_file_type(filename,mode))) goto ErrorDetectFormat;
  }
  TRY(ctx=g_formats[ifmt]->open(filename,mode));
  NEW(struct _ndio_t,file,1);
  memset(file,0,sizeof(struct _ndio_t));
  file->context=ctx;
  file->fmt=g_formats[ifmt];
  return file;
ErrorSpecificFormat:
  LOG("%s(%d): %s"ENDL "\tCould not open \"%s\" for %s with specified format %s."ENDL,
      __FILE__,__LINE__,__FUNCTION__,filename?filename:"(null)",(mode[0]=='r')?"reading":"writing",format);
  return NULL;
ErrorDetectFormat:
  LOG("%s(%d): %s"ENDL "\tCould not detect file format of \"%s\" for %s."ENDL,
      __FILE__,__LINE__,__FUNCTION__,filename?filename:"(null)",(mode[0]=='r')?"reading":"writing"); 
  return NULL;
Error:
  return NULL;
}

/** Opens a file according to the mode.

    \param[in] filename_fmt The path to the file. printf()-style formating is used
                            to format the filename.
    \param[in] format   The name of the desired format, or NULL.  If NULL,
                        an attempt will be made to infer the file format
                        from the file itself (if present) or the filename.
                        If no matching format is found, a default format
                        will be used to write the file(mode w), but a read
                        (mode r) will fail.
    \param[in] mode     May be "r" to open the file for reading or "w" to
                        open the file for writing.

    \returns NULL on failure, otherwise an ndio_t object.
 */
ndio_t ndioOpenv (const char *filename_fmt, const char *format, const char *mode, ...)
{ char buf[1024]={0};
  va_list args;
  va_start(args,mode);
  vsnprintf(buf,sizeof(buf),filename_fmt,args);
  va_end(args);
  return ndioOpen(buf,format,mode);
}

/** Closes the file and releases resources.  Always succeeds. */
void ndioClose(ndio_t file)
{ if(!file) return;
  file->fmt->close(file);
  ndfree(file->shape);
  ndfree(file->cache);
  SAFEFREE(file->dstpos);
  SAFEFREE(file->srcpos);
  SAFEFREE(file->seekable)

  if(file->log) fprintf(stderr,"Log: 0x%p"ENDL "\t%s"ENDL,file,file->log);
  SAFEFREE(file->log);
  free(file);
}

nd_t ndioShape(ndio_t file)
{ return file?file->fmt->shape(file):0;
}

/// @cond DEFINES
#undef LOG
#define LOG(...) ndLogError(a,__VA_ARGS__)
/// @endcond

ndio_t ndioRead(ndio_t file, nd_t a)
{ TRY(file);
  TRY(file->fmt->read(file,a));
  return file;
Error:
  if(ndioError(file))
    ndLogError(a,"[nD IO Error]"ENDL "%s"ENDL,ndioError(file));
  return NULL;
}

ndio_t ndioWrite(ndio_t file, nd_t a)
{ nd_t t=0;
  ndio_t out=file;
  TRY(file);
  TRY(a);
  if(ndkind(a)==nd_gpu_cuda)
  { TRY(ndcopy(t=ndheap(a),a,0,0));
    a=t; 
  }
  TRY(file->fmt->write(file,a));
Finalize:
  ndfree(t);
  return file;
Error:
  out=NULL;
  if(ndioError(file))
    ndLogError(a,"[nD IO Error]"ENDL "%s"ENDL,ndioError(file));
  goto Finalize;
}

/// @cond DEFINES
#undef LOG
#define LOG(...) ndioLogError(file,__VA_ARGS__)
/// @endcond

/** Set format specific data.
 *
 *  \param[in] file   An open file.
 *  \param[in] param  A buffer of size \a nbytes.  The contents required
 *                    depend on the specific format of the file.
 *  \param[in] nbyets The number of bytes in the \a param buffer.
 *  \returns 0 on failure, otherwise \a file.
 */
ndio_t ndioSet(ndio_t file, void *param, size_t nbytes)
{ TRY(file);
  TRY(file->fmt->set); // some formats may not implement set()
  TRY(file->fmt->set(file,param,nbytes));
Error:
  return NULL;
}

/**
 * \param[in] An ndio_t object.
 * \returns The name of the plugin used to read/write \a file,
 *          or the string "(error)" if there was an error.
 */
const char* ndioFormatName(ndio_t file)
{ TRY(file);
  TRY(file->fmt->name);
  return file->fmt->name();
Error:
  return "(error)";
}

/** Set format specific data.
 *
 *  \param[in]  file   An open file.
 *  \returns 0 on failure, otherwise a pointer to format specific data.
 */

void* ndioGet(ndio_t file)
{ TRY(file);
  TRY(file->fmt->get); // some formats may not implement get()
  return file->fmt->get(file);
Error:
  return NULL;
}
#undef LOG

/** Appends message to error log for \a file
    and prints it to \c stderr.
  */
void ndioLogError(ndio_t file,const char *fmt,...)
{ size_t n,o;
  va_list args,args2;
  if(!file) goto Error;
  va_start(args,fmt);
  va_copy(args2,args);      // [HAX] this is pretty ugly
#ifdef _MSC_VER
  n=_vscprintf(fmt,args)+1; // add one for the null terminator
#else
  n=vsnprintf(NULL,0,fmt,args)+1;
#endif
  va_end(args);
  if(!file->log) file->log=(char*)calloc(n,1);
  //append
  o=strlen(file->log);
  file->log=(char*)realloc(file->log,n+o);
  memset(file->log+o,0,n);
  if(!file->log) goto Error;
  va_copy(args,args2);
  vsnprintf(file->log+o,n,fmt,args2);
  va_end(args2);
#if 0
  vfprintf(stderr,fmt,args); // mirror output to stderr
#endif
  va_end(args);
  return;
Error:
  va_start(args2,fmt);
  vfprintf(stderr,fmt,args2); // print to stderr if logging fails.
  va_end(args2);
  fprintf(stderr,"%s(%d): Logging failed."ENDL,__FILE__,__LINE__);
}

void ndioResetLog(ndio_t file) {SAFEFREE(file->log);}

//
// === READ SUBARRAY ===
//

/// @cond DEFINES
#define LOG(...)               ndioLogError(file,__VA_ARGS__)
#define min_(a,b)              (((a)<(b))?(a):(b))
#define max_(a,b)              (((a)<(b))?(b):(a))
#define step_(idx_)            (step?step[idx_]:1)
#define ori_(idx_)             (origin?origin[idx_]:0)
// If a file format doesn't support seeking, the entire array is read into cache
// These macros replace the format implementations with defaults to effect this 
// behavior.
#define canseek_(file_,i_)     (file_->fmt->canseek?file_->fmt->canseek(file_,i_):0)
#define seek_(file_,vol_,pos_) (file_->fmt->seek?file_->fmt->seek(file_,vol_,pos_):file_->fmt->read(file_,vol_))
// Utility memory ops
#define MAYBE_REALLOC(T,ptr,n) do{\
                            if(n>ptr##_n)\
                            { ptr##_n=n;\
                              TRY(ptr=(T*)realloc(ptr,sizeof(T)*(ptr##_n)));\
                            }\
                          } while(0)
#define ZERO(T,ptr,n)     memset(ptr,0,sizeof(T)*(ptr##_n))
/// @endcond

/**
 * Vector increment of \a pos in \a domain with carry.
 * Only increment dimensions with a corresonding 1 in \a mask.
 * Dimensions with mask==0, have an effective shape of 1.
 *
 * Each call is ~O(ndndim(domain)).
 */
static unsigned inc(nd_t domain,size_t *pos, char *mask)
{ unsigned kdim=0;//=ndndim(domain)-1;
  while(kdim<ndndim(domain) && (!mask[kdim] || pos[kdim]==ndshape(domain)[kdim]-1))
    pos[kdim++]=0;
  if(kdim>=ndndim(domain)) return 0;
  pos[kdim]++;
#if 0
  { size_t i;
    printf("ndioReadSubarray:inc(376) :: ");
    for(i=0;i<ndndim(domain);++i)
      printf("%5zu",pos[i]);
    printf(ENDL);
  }
#endif  
  return 1;
}
/// (for subarray) set offset for sub-array relative to \a ori
static void setpos(nd_t src,const size_t *ipos, size_t *ori)
{ size_t i;  
  for(i=0;i<ndndim(src);++i)
    ndoffset(src,(unsigned)i,((int64_t)ipos[i])-(ori?ori[i]:0));
}
/// (for subarry) Undo setpos() by negating the offset for a sub-array
static void unsetpos(nd_t src,const size_t *ipos, size_t *ori)
{ size_t i;
  for(i=0;i<ndndim(src);++i)
    ndoffset(src,(unsigned)i,-(int64_t)ipos[i]+(ori?ori[i]:0));
}
/** Compute: out=o+p*s
 *  For subarray.
 *  \param[out]   out    output position vector with \a nd elements.
 *  \param[in]    p      input position vector with at least \a mind elements.
 *                       Positions past \a mind are treated as zeros.
 *  \param[in]    origin origin vector in \a out space with \a nd elements.
 *  \param[in]    step   step size vector in \a out space with \a nd elements.
 */
static void getsrcpos(size_t mind, size_t nd, size_t *out, size_t *p, size_t *origin, size_t *step)
{ size_t i;
  for(i=0;i<mind;++i)
    out[i]=ori_(i)+p[i]*step_(i);
  for(;i<nd;++i)
    out[i]=ori_(i);
}
/// \returns 1 if file's srcpos is not in cache
static unsigned cachemiss(ndio_t file)
{ size_t i;
  const size_t *sp=file->srcpos,
               *sh=ndshape(file->cache),
               *cp=file->cachepos;
  if(!cp) return 1;
  for(i=0;i<ndndim(file->shape);++i)
    if(sp[i]<cp[i] || (cp[i]+sh[i])<=sp[i]) // hit if sp in [cp,cp+sh)
      return 1;
  return 0;
}

/** Read a sub-volume from a file.
 *  Usage:
 *  \code
 *  { ndiot_t file=ndioOpen("blah",0,"r"); // write mode supports append...doesn't need to support slicing
 *    nd_t vol=ndioShape(file);
 *    size_t n;
 *    // Assume we know the dimensionality of our data and which dimension to iterate over.
 *    n=ndshape()[2];      // remember the range over which to iterate
 *    ndShapeSet(vol,2,1); // prep to iterate over 3'rd dimension (e.g. expect WxHxDxC data, read WxHx1XC planes)
 *    ndref(vol,malloc(ndnbytes(vol)),nd_heap); // alloc just enough data      
 *    { int64_t pos[]={0,0,0,0}; // 4d data
 *      size_t i;
 *      for(i=0;i<n;++i,++pos[2])
 *      { ndioReadSubarray(file,vol,pos,0); // seek to pos and read, shape limited by vol
 *        f(vol);                           // do something with the result
 *      }
 *    }
 *    ndfree(vol);
 *    ndioClose(file);
 *  }
 *  \endcode
 *  \param[in]      file    A ndio_t object opened with read mode.
 *  \param[in,out]  dst     The destination array. Must have valid data pointer.
 *                          The kind should be compatible with ndcopy().
 *                          The read will attempt to fill the specified shape.
 *                          The domain requested must fit within the shape of 
 *                          the array described by \a file.
 *  \param[in]      origin  An array of <tt>ndndim(ndioShape(file))</tt> 
 *                          elements.  This point in the file will correspond to
 *                          (0,...) in the \a dst array.  If NULL, the origin 
 *                          will be set to (0,...).
 *  \param[in]      step    An array of <tt>ndndim(ndioShape(file))</tt>
 *                          elements that specifies the step size to be taken
 *                          along each dimension as it is read into \a dst.
 *                          If NULL, a step size of 1 on each dimension will be 
 *                          used.
 *  \see _ndio_fmt_t.subarray
 *  \see _ndio_fmt_t.seek
 *  \see _ndio_fmt_t.canseek
 */
ndio_t ndioReadSubarray(ndio_t file, nd_t dst, size_t *origin, size_t *step)
{ // need to know minimum seekable dimension
  // maximum non-seekable dimensions
  size_t ndim,max_unseekable=0;  /// \todo do i use max_unseekable?
  unsigned use_cache=0;
  void *ref=nddata(dst); // remember the original pointer so nddata(dst) doesn't change even when this call fails  
  
  // Check for direct format support
  if(file->fmt->subarray)
  { size_t *ori_=origin,*step_=step;
    if(!ori_) // Handle origin is NULL
    { size_t i;
      ori_ =(size_t*)alloca(ndndim(dst)*sizeof(size_t));
      for(i=0;i<ndndim(dst);++i)
        ori_[i]=0;
    }
    if(!step_) // Handle step is NULL
    { size_t i;
      step_ =(size_t*)alloca(ndndim(dst)*sizeof(size_t));
      for(i=0;i<ndndim(dst);++i)
        step_[i]=1;
    }
    if(file->fmt->subarray(file,dst,ori_,step_))
      return file;
    else
      return 0;
  }
  /*
    File format doesn't directly support subarray() interface.
    There are two other options: 
      (A) Format supports seek()/canseek() or 
      (B) Format only permits reading the whole volume.
    In the case of (B) the entire volume is read into an internal cache on the
    first ndioReadSubarray() call.  Subsequent calls use the cache.

    In the case of (A) the subvolume is pieced together from certain subvolumes.
    It's assumed that a seekable dimension supports reading a hyperplane
    (shape 1 on that dimension) from a given location.  For non-seekable
    dimensions, the entire dimension must be read.

    For example, video formats might support seeking of images.  Any time point
    can be addressed at (amortized) constant time, but an entire 2d image must
    be read at that time point.

    Selecting a subvolume from a "seekable" format might also incorporate use of
    an in-memory cache, but will use information about which dimensions are
    seekable to minimize file access.
   */

  if(!file->shape) TRY(file->shape=ndioShape(file));
  ndim=min_(ndndim(file->shape),ndndim(dst));
  MAYBE_REALLOC(char,file->seekable,ndndim(dst));

  // Check for out-of-bounds request
  // Reading a box of shape dst starting at the origin.  The origin specified in units specified by the step size.
  { size_t i;
    for(i=0;i<ndim;++i)
      TRY((ndshape(dst)[i]+ori_(i)*step_(i))<=ndshape(file->shape)[i]); // spell it out for the error message
    for(;i<ndndim(file->shape);++i)  // rest of the dst shape==1
      TRY(1+ori_(i)<=ndshape(file->shape)[i]);
  }

  // Need to cache a dim if it's not seekable and shape[i]<dst->shape[i]
  // Only need to cache up to maximum unseekable dim (dims>ndim are treated as seekable).
  
  // First check for the need to cache  
  { size_t i;
    const size_t *dsh=ndshape(dst),
                 *fsh=ndshape(file->shape);
    for(i=0;i<ndim;++i)
    { if(!(file->seekable[i]=canseek_(file,i)))
      { max_unseekable=i;
        if(dsh[i]<fsh[i]) // then need to cache dim
          use_cache=1;
      }
    }
    for(;i<ndndim(dst);++i)
      file->seekable[i]=1;
  }

  // Chack to see if cache needs resizing
  if(use_cache)
  { size_t i;
    const size_t *dsh=ndshape(dst),
                 *fsh=ndshape(file->shape);
    if(!file->cache)
      TRY(file->cache=ndunknown(file->shape));
    for(i=0;i<=max_unseekable;++i)
    { if(i>ndim || canseek_(file,i)) // other dims get full size
        ndShapeSet(file->cache,(unsigned)i,1); //may insert dimensions
    }
    for(;i<ndndim(file->cache);++i)  //set shape of any remaining dimensions
      ndShapeSet(file->cache,(unsigned)i,1);
    // (re)alloc cache
    TRY(ndref(file->cache,
              realloc(nddata(file->cache),ndnbytes(file->cache)),
              nd_heap));
  }

  // Allocate and init position indexes
  MAYBE_REALLOC(size_t,file->dstpos,ndndim(dst));
  ZERO(size_t,file->dstpos,ndndim(dst));
  MAYBE_REALLOC(size_t,file->srcpos,ndndim(file->shape));

  // Read
  if(!use_cache)
  { do
    { setpos(dst,file->dstpos,0);
      getsrcpos(ndim,ndndim(file->shape),file->srcpos,file->dstpos,origin,step); // srcpos=origin+dstpos*step
      TRY(seek_(file,dst,file->srcpos)); // read directly to dst
      unsetpos(dst,file->dstpos,0);
#if 0
      printf("ndim: %2d\t",(int)ndim);
      { int i;
        for(i=0;i<ndim;++i) printf(" %5d",(int)file->dstpos[i]);
      }
      printf("\n"); fflush(stdout);
      //if(ndim==3) ndioClose(ndioWrite(ndioOpenv("%d.tif",NULL,"w",(int)file->srcpos[2]),dst));
      if(ndim==5)   ndioClose(ndioWrite(ndioOpenv("%d.%d.h5",NULL,"w",(int)file->srcpos[2],(int)file->srcpos[4]),dst));
#endif
    } while(inc(dst,file->dstpos,file->seekable));
  } else
  { do
    { getsrcpos(ndim,ndndim(file->shape),file->srcpos,file->dstpos,origin,step); // srcpos=origin+dstpos*step
      if(cachemiss(file))
      { TRY(seek_(file,file->cache,file->srcpos));
        MAYBE_REALLOC(size_t,file->cachepos,ndndim(file->shape));
        memcpy(file->cachepos,file->srcpos,sizeof(size_t)*ndndim(file->shape));
      }
      
      setpos(dst,file->dstpos,0);
      setpos(file->cache,origin,file->cachepos);
      TRY(ndcopy(dst,file->cache,ndim,ndshape(dst)));//cache has file's dimensionality.
      unsetpos(file->cache,origin,file->cachepos);
      unsetpos(dst,file->dstpos,0);
    } while(inc(dst,file->dstpos,file->seekable));
  }

  return file;
Error:
  ndref(dst,ref,ndkind(dst)); // restore original nddata(dst) in case it was modified
  if(ndioError(file)) // copy file error log to the array's log
    ndLogError(dst,"[nD IO Error]"ENDL "%s"ENDL,ndioError(file));
  return 0;
}

/**
 * Query whether the file format supports seeking along dimension \a idim.
 * This can be used to guide calls to ndioReadSubarray() whose cacheing 
 * behavior depends on which dimensions are seekable.
 *
 * However, normally, you shouldn't need to call this function.  It is included
 * in the public interface mostly to aid in implementing new ndio formats. 
 */
unsigned ndioCanSeek(ndio_t file, size_t idim)
{ return canseek_(file,idim);
}

#pragma warning( pop )
