#include "plugin.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <malloc.h>

#pragma warning( push )
#pragma warning( disable:4996 ) //unsafe function

//#define DEBUG_SEARCH

#define ENDL       "\n"
#define LOG(...)   fprintf(stderr,__VA_ARGS__)
#define TRY(e,msg) do{ if(!(e)) {LOG("%s(%d): %s"ENDL "\tExpression evaluated to false."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,msg); goto Error; }} while(0)
#define SILENTTRY(e,msg) do{ if(!(e)) { goto Error; }} while(0)

#ifdef _MSC_VER
#include <windows.h>
#include "dirent.win.h"                                     // use posix-style directory traversal
const char* estring();
#else // POSIX
#include <dlfcn.h>
#include <dirent.h>
//For dynamic loading: alias posix calls to the windows calls.
#define LoadLibrary(name)        dlopen((name),RTLD_LAZY)   // returns NULL on error, use dlerror()
#define FreeLibrary(lib)         (0==dlclose(lib))          // dlclose returns non-zero on error, use dlerror().  FreeLibrary returns 0 on error, otherwise non-zero
#define GetProcAddress(lib,name) dlsym((lib),(name))        // returns NULL on error, use dlerror()
#define estring                  dlerror                    // returns an error string
#define HMODULE                  void*
#endif

#ifdef _MSC_VER
#define EXTENSION "dll"
#elif defined(__APPLE__) && defined(__MACH__)
#define EXTENSION "dylib"
#else
#define EXTENSION "so"
#endif

static size_t min_sz_t(size_t a, size_t b) {return (a<b)?a:b;}

static int is_shared_lib(const char *fname,size_t n)
{ const char *dot;
  TRY(dot=strrchr(fname,'.'),"No extension found in file name.");
  return 0==strncmp(dot+1,EXTENSION,min_sz_t(n-(dot-fname+2),sizeof(EXTENSION)));
Error:
  return 0;
}

static void cat(char *buf, size_t nbytes, int nstrings, const char** strings)
{ int i;
  memset(buf,0,nbytes);
  for(i=0;i<nstrings;++i)
    strcat(buf,strings[i]);
}

static ndio_fmt_t *load(const char *path, const char *fname)
{ ndio_fmt_t *api=NULL;
  void *lib=NULL;
  ndio_get_format_api_t get;
  char *buf;
  const char *p[]={(char*)path,"/",(char*)fname}; // windows handles the "/" just fine
  size_t n = strlen(path)+strlen(fname)+2; // one extra for the terminating null
  TRY(buf=(char*)alloca(n),"Out of stack space.");
  cat(buf,n,3,p);
  /// \todo FIXME: WINDOWS SPECIFIC (SetDllDirectory)
  // - add the path to the loaded plugin as to the shared lib search path so that
  //   shared libaries that the plugin depends on will also be found.
  // - not sure what the linux behavior is so not sure how to fix this yet
  TRY(SetDllDirectory(path),estring());
  TRY(lib=LoadLibrary(fname),"There was a problem loading the specified library.");
  SILENTTRY(get=(ndio_get_format_api_t)GetProcAddress((HMODULE)lib,"ndio_get_format_api"),estring());
  TRY(api=(ndio_fmt_t*)get(),fname);

  api->lib=lib;
Finalize:
  return api;
Error:
  if(lib) { FreeLibrary((HMODULE)lib); lib=NULL; }
  goto Finalize;
}

typedef struct _apis_t
{ ndio_fmt_t **v;
  size_t       n,cap;
} apis_t;

static int push(apis_t *a, ndio_fmt_t *fmt)
{ ndio_fmt_t **v=a->v;
  size_t n      =a->n,
         cap    =a->cap;
  if(!fmt)
    return 1; //ignore with success
  if(n+1>=cap)
    TRY(v=(ndio_fmt_t**)realloc(v,sizeof(*v)*( cap=(size_t)(1.2*n+10) )),"Expanding format API array.");
  v[n++]=fmt;
  a->v=v;
  a->n=n;
  a->cap=cap;
  return 1;
Error:
  return 0;
}


static int recursive_load(apis_t *apis,DIR* dir,const char *path)
{ DIR* child=0;
  int is_ok=1;
  struct dirent *ent=0;
  while((ent=readdir(dir))!=NULL)
  { if(ent->d_type==DT_REG
    && is_shared_lib(ent->d_name,strlen(ent->d_name)))
    { TRY(push(apis,load(path,ent->d_name)),"Could not append format API.");
    } else if(ent->d_type==DT_DIR)
    { char *buf;
      size_t n=strlen(path)+strlen(ent->d_name)+2;
      const char *p[]={path,"/",ent->d_name};
      if( 0==strcmp(ent->d_name,".")*strcmp(ent->d_name,".."))
        continue;
      TRY(buf=(char*)alloca(n),"Out of stack space.");
      cat(buf,n,3,p);
#ifdef DEBUG_SEARCH
      puts(buf);
#endif
      TRY(child=opendir(buf),"Could not open child directory.");
      TRY(recursive_load(apis,child,buf),"Search for plugins failed.");
    }
  }
Finalize:
  if(child) closedir(child);
  return is_ok;
Error:
  is_ok=0;
  goto Finalize;
}

ndio_fmts_t ndioLoadPlugins(const char *path, size_t *n)
{ apis_t apis = {0};
  DIR*           dir;
  TRY(dir=opendir(path),strerror(errno));
  TRY(recursive_load(&apis,dir,path),"Search for plugins failed.");
  *n=apis.n;
Finalize:
  closedir(dir);
  return apis.v;
Error:
  if(n) *n=0;
  apis.v=NULL;
  goto Finalize;
}

void ndioFreePlugins(ndio_fmts_t fmts, size_t n)
{ size_t i;
  if(!fmts) return;
  for(i=0;i<n;++i)
  { if(fmts[i] && fmts[i]->lib)
      TRY(FreeLibrary((HMODULE)fmts[i]->lib),estring());
Error:
    ; // keep trying to free the others
  }
  free(fmts);
}

#ifdef _MSC_VER
/** \todo make thread safe...add a mutex */
static
const char* estring()
{ static char buf[4096];
  FormatMessage(
      FORMAT_MESSAGE_FROM_SYSTEM |
      FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL,
      GetLastError(),
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPTSTR) &buf,
      sizeof(buf)/sizeof(TCHAR), NULL);
  return buf;
}
#endif

#pragma warning( pop )
