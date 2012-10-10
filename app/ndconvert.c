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

#ifdef _MSC_VER
#define PATHSEP '\\'
#else
#define PATHSEP '/'
#endif

typedef struct _opt_t
{ char *srcname;
  char *dstname;
} *opt_t;

char* basename(char* path)
{ char *n=strrchr(path,PATHSEP);
  return n?(n+1):path;
}

void usage(char *name)
{ printf("Usage:\n\t%s <source-file-name> <destination-file-name>\n\t%s\n",basename("ndconvert"),name);
}

int optparse(opt_t opts, int argc,char* argv[])
{ if(argc!=3)
    goto Error;
  opts->srcname=argv[1];
  opts->dstname=argv[2];
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
  nd_t a;
  int ecode=0;
  init();
  if(!optparse(&opts,argc,argv)) return -1;
  src=ndioOpen(opts.srcname,0,"r");
  dst=ndioOpen(opts.dstname,0,"w");
  a=ndioShape(src);
  ndref(a,malloc(ndnbytes(a)),nd_heap);
  ndioClose(ndioRead(src,a));
  ndioClose(ndioWrite(dst,a));
Finalize:
  ecode=nderror(a)!=NULL;
  ndfree(a);
  return ecode;
}
