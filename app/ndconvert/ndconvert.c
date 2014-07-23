/**
 * \file
 * A command line application for converting between supported data formats.
 *
 * This tool is a bit of a testbed for features in the nd/ndio API.
 *
 * \todo FIXME Some formats do not support all pixel types...how to specify when
 *       a conversion is required?
 * 
 * \todo Add command line supportt for specifying source and destination formats
 * \todo Use niodReadSubarray so that the entire data set doesn't have to be
 *       read into memory.
 * \todo setup producer/consumer threads so that one thread can be
 *       reading while the other is writing.
 * \todo dimension annotation so color dimensions get identified properly
 * \todo (limitation) when possible persist frame rate information for ffmpeg transcoding?
 */
#include "nd.h"
#include "config.h"
#include <string.h>
#include <stdio.h>

#define countof(e) (sizeof(e)/sizeof(*(e)))

#ifdef _MSC_VER
#define PATHSEP '\\'
#else
#define PATHSEP '/'
#endif

typedef struct _opt_t                 
{ char *srcname;
  char *dstname;
  nd_type_id_t type;
} *opt_t;

char* basename(char* path)
{ char *n=strrchr(path,PATHSEP);
  return n?(n+1):path;
}

  struct {nd_type_id_t tid; const char *name; const char *desc;} type_desc[]={
    {nd_id_unknown,"Type","Description"},
    {nd_id_unknown,"-----","----------------------------"},
    {nd_u8 ,       "u8" ,"      unsigned 8-bit"},
    {nd_u16,       "u16","      unsigned 16-bit"},
    {nd_u32,       "u32","      unsigned 32-bit"},
    {nd_u64,       "u64","      unsigned 64-bit"},
    {nd_i8 ,       "i8" ,"        signed 8-bit"},
    {nd_i16,       "i16","        signed 16-bit"},
    {nd_i32,       "i32","        signed 32-bit"},
    {nd_i64,       "i64","        signed 64-bit"},
    {nd_f32,       "f32","floating-point 32-bit"},
    {nd_f64,       "f64","floating-point 64-bit"}};

void usage(char *name)
{ int i=0;
  printf("Usage:\n\t%s <source-file-name> <destination-file-name> [pixel-type]\n",basename(name));
  printf("\tValid pixel types are:\n");
  for(i=0;i<countof(type_desc);++i)
    printf("\t\t%5s\t%-40s\n",type_desc[i].name,type_desc[i].desc);
  printf("\tSome pixel types do not work with certain output formats.\n");
  printf("\n");
}

unsigned streq(const char *a, const char *b)
{ if(strlen(a)!=strlen(b)) return 0;
  return strcmp(a,b)==0;
}

nd_type_id_t parse_type(const char* typestring)
{ int i;
  for(i=2;i<countof(type_desc);++i)
    if(streq(typestring,type_desc[i].name))
      return type_desc[i].tid;
  return nd_id_unknown;
}

int optparse(opt_t opts, int argc,char* argv[])
{ if(argc<3 || 4<argc)
    goto Error;
  opts->srcname=argv[1];
  opts->dstname=argv[2];
  opts->type=nd_id_unknown;
  if(argc==4) // specified type
    if( (opts->type=parse_type(argv[3])) == nd_id_unknown)
      goto Error;
  return 1;
Error:
  usage(argv[0]);
  return 0;
}

void init()
{ ndioAddPluginPath(NDIO_BUILD_ROOT); // add build root so we can successfully run the program from an msvc build system
}

int main(int argc,char* argv[])
{ struct _opt_t opts;
  ndio_t src,dst;
  nd_t a,b;
  int ecode=0;
  init();
  if(!optparse(&opts,argc,argv)) return -1;
  src=ndioOpen(opts.srcname,0,"r");
  dst=ndioOpen(opts.dstname,0,"w");
  a=ndioShape(src);
  b=ndref(a,malloc(ndnbytes(a)),nd_heap);
  ndioClose(ndioRead(src,b));
  if(opts.type!=nd_id_unknown)
    b=ndconvert_ip(b,opts.type);
  ndioClose(ndioWrite(dst,b));
Finalize:
  ecode=nderror(a)!=NULL;
  ndfree(a);
  return ecode;
}
