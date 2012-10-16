/**
 * \file
 * An ndio plugin for writing raw data.
 *
 * \todo Append raw data
 *
 * \author Nathan Clack
 * \date   Aug 2012
 */
#include "nd.h"
#include <stdio.h>
/// @cond DEFINES
#define countof(e) (sizeof(e)/sizeof(*e))
#define ENDL                  "\n"
#define LOG(...)              printf(__VA_ARGS__)
#define TRY(e)                do{if(!(e)) { LOG("%s(%d): %s()"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error;}} while(0)
#define TRYMSG(e,msg)         do{if(!(e)) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL "\t%sENDL",__FILE__,__LINE__,__FUNCTION__,#e,msg); goto Error; }}while(0)
#define FAIL(msg)             do{ LOG("%s(%d): %s()"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,msg); goto Error;} while(0)
#define RESIZE(type,e,nelem)  TRY((e)=(type*)realloc((e),sizeof(type)*(nelem)))
#define NEW(type,e,nelem)     TRY((e)=(type*)malloc(sizeof(type)*(nelem)))
#define SAFEFREE(e)           if(e){free(e); (e)=NULL;}
/// @endcond
//
//  === HELPERS ===
//
//
/**
 * Set read/write mode flags according to mode string.
 */
static
unsigned parse_mode_string(const char* mode)
{ char *c;
  for(c=(char*)mode;*c;++c)
  { if(*c!='w') FAIL("Invalid mode string");
  }
  return 1;
Error:
  return 0;
}
/** The format name.
    Use the format name to select this format.
*/
static const char* raw_fmt_name(void) { return "raw"; }
/** This format is disabled for autodetection.
    \returns 0.
 */
static unsigned raw_is_fmt(const char* path, const char *mode)
{ return 0;
}
/**
 * Opens a raw file.  Only writing permitted.
 */
static void* raw_open(const char* path, const char* mode)
{ FILE *out=0;
  TRY(parse_mode_string(mode));
  TRY(out=fopen(path,mode));
  return out;
Error:
  LOG("%s(%d): %s()"ENDL "\tCould not open"ENDL "\t\t%s"ENDL "\t\twith mode \"%s\""ENDL,
      __FILE__,__LINE__,__FUNCTION__,path?path:"(null)",mode?mode:"(null)");
  return 0;
}
/** Releases resources */
static void raw_close(ndio_t file)
{ FILE *self=(FILE*)ndioContext(file);
  if(self) fclose(self);
}
/**
 * No read support. \returns 0.
 */
static nd_t raw_shape(ndio_t file)
{ return 0;
}
/**
 * No read support. \returns 0.
 */
static unsigned raw_read(ndio_t file,nd_t dst)
{ return 0;
}
/**
 * Write a file raw.
 */
static unsigned raw_write(ndio_t file, nd_t src)
{ FILE *self=(FILE*)ndioContext(file);
  TRY(ndnelem(src)==fwrite(nddata(src),ndbpp(src),ndnelem(src),self));
  return 1;
Error:
  return 0;
}
/**
 * Seek. No read support. \returns 0.
 */
static unsigned raw_seek(ndio_t file, nd_t dst, size_t *pos)
{ return 0;
}
/**
 * Query which dimensions ares seekable.
 * No read support. \returns 0.
 */
static unsigned raw_canseek(ndio_t file, size_t idim)
{ return 0;
}
//
// === EXPORT ===
//
/// @cond DEFINES
#ifdef _MSC_VER
#define shared __declspec(dllexport)
#else
#define shared
#endif
/// @endcond
#include "src/io/interface.h"
/// Interface function for the ndio-raw plugin.
shared const ndio_fmt_t* ndio_get_format_api(void)
{
  static ndio_fmt_t api=
  { raw_fmt_name,
    raw_is_fmt,
    raw_open,
    raw_close,
    NULL,//raw_shape,
    NULL,//raw_read,
    raw_write,
    NULL, //set
    NULL, //get
    NULL,//raw_canseek,
    NULL,//raw_seek,
    NULL
  };
  return &api;
}