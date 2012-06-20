#include "nd.h"
#include "io.h"
#include "io/interface.h"
#include "image.h"
#include <string.h>

#define ENDL              "\n"
#define LOG(...)          fprintf(stderr,__VA_ARGS__)
#define TRY(e)            do{if(!(e)) { LOG("%s(%d):"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,#e); goto Error;}} while(0)
#define NEW(type,e,nelem) TRY((e)=(type*)malloc(sizeof(type)*(nelem)))
#define SAFEFREE(e)       if(e){free(e); (e)=NULL;}
#define FAIL              do{ LOG("Execution should not have reached this point."ENDL); goto Error; }while(0)

static unsigned islsm(const char* path)
{ return 0==strcmp(strrchr(path,'.'),"lsm");
}

/////
///// INTERFACE
/////

static const char* name() { return "tiff/mylib"; }

static unsigned is(const char *path, const char *mode)
{ Tiff *t=Open_Tiff(path,mode));
  if(t) Close_Tiff(t);
  return t!=NULL;
}

static void* open(const char* path, const char *mode)
{ return Open_Tiff(path,mode);
}

// Following functions will log to the file object.
#undef  LOG
#define LOG(...) ndioLogError(file,__VA_ARGS__)

static void close(ndio_t file)
{ Tiff *ctx;
  if(!file) return;
  ctx=(tiff_ctx_t*)ndioContext(file);
  if(ctx) Tiff_Close(ctx);
}

static int pack(size_t *s, int n)
{ int i,c;
  for(i=0,c=0;i<n;++i)
  { s[c]=s[i];
    if(s[i]!=1) ++c;
  }
  return i;
}

static size_t prod(const size_t *s, int n)
{ size_t i,p=1;
  for(i=0;i<n;++i) p*=s[i];
  return p;
}

static nd_t shape(ndio_t file)
{ 
  int w,h,c,d;
  Tiff *ctx;
  TRY(ctx=(tiff_ctx_t*)ndioContext(file));
  // assumes first ifd is representative
  Rewind_Tiff(ctx); // just in case? I don't know if there's a way to remember the ifd we we're on 
  TRY(0==Get_IFD_Shape(ctx,&w,&h,&c));
  for(d=0;!Tiff_EOF(ctx);Advance_Tiff(ctx),++d); // count planes
  Rewind_Tiff(ctx);
  { nd_t out=ndinit();
    size_t k,shape[]={w,h,d,c};
    k=pack(shape,n);    
    ndref(out,NULL,prod(s,k));
    ndreshape(out,k,s);
    return out;
  }
Error:
  { char* msg;
    if(msg=Image_Error())
    { LOG("%s(%d): Image Error"ENDL "\t%s"ENDL,__FILE__,__LINE__,msg);
      Image_Error_Release();
    }
  }
  return 0;
}

/////
///// EXPORT
/////

#define shared
#ifdef _MSC_VER
#define shared __declspec(dllexport)
#endif

shared const ndio_fmt_t* ndio_get_format_api(void)
{ static ndio_fmt_t api = {0};
  api.name   = name;
  api.is_fmt = is;
  api.open   = open;
  api.close  = close;
  api.shape  = shape;
  api.read   = read;
  api.write  = write;
  return &api;
}
