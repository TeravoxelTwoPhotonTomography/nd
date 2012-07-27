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
#include "config.h"
#include "io/plugin.h"
#include "io/interface.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#ifdef _MSC_VER
#define va_copy(a,b) (a=b)
#endif

/// @cond DEFINES
#define ENDL                         "\n"
#define LOG(...)                     fprintf(stderr,__VA_ARGS__)
#define TRY(e)                       do{if(!(e)) { LOG("%s(%d): %s"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error;}} while(0)
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
  char       *log;       ///< Used to store the error log.  NULL if no errors, otherwise a NULL terminated string.
};

/////
///// HELPERS
/////

/** \todo make thread safe, needs a mutex */
static int maybe_load_plugins()
{ if(!g_formats)
    TRY(g_formats=ndioLoadPlugins(NDIO_PLUGIN_PATH,&g_countof_formats));
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

/////
///// INTERFACE
/////

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
  if(format)
  { if(0>(ifmt=get_format_by_name(format))) goto ErrorSpecificFormat;
  } else
  { if(0>(ifmt=detect_file_type(filename,mode))) goto ErrorDetectFormat;
  }
  TRY(ctx=g_formats[ifmt]->open(filename,mode));
  NEW(struct _ndio_t,file,1);
  file->context=ctx;
  file->fmt=g_formats[ifmt];
  file->log=NULL;
  return file;
ErrorSpecificFormat:
  LOG("%s(%d): %s"ENDL "\tCould not open \"%s\" for %s with specified format %s."ENDL,
      __FILE__,__LINE__,__FUNCTION__,filename?filename:"(null)",(mode[0]=='r')?"reading":"writing",format);
  return NULL;
ErrorDetectFormat:
  LOG("%s(%d): %s"ENDL "\tCould not open \"%s\" for %s."ENDL,
      __FILE__,__LINE__,__FUNCTION__,filename?filename:"(null)",(mode[0]=='r')?"reading":"writing"); 
  return NULL;
Error:
  return NULL;
}

/** Closes the file and releases resources.  Always succeeds. */
void ndioClose(ndio_t file)
{ if(!file) return;
  file->fmt->close(file);
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
{ TRY(file);
  TRY(a);
  TRY(file->fmt->write(file,a));
  return file;
Error:
  if(ndioError(file))
    ndLogError(a,"[nD IO Error]"ENDL "%s"ENDL,ndioError(file));
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

#pragma warning( pop )
