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
#include "opts.h"
#include "config.h"
#include <string.h>
#include <stdio.h>

#define countof(e) (sizeof(e)/sizeof(*(e)))

#define TRY(e) do{ if(!(e)) goto Error; } while(0)

int doone(const char* s, const char *d, nd_type_id_t t)
{ int isok=1;
  ndio_t src,dst;
  nd_t a,b; // b just references a...tracks error state
  src=ndioOpen(s,0,"r");
  dst=ndioOpen(d,0,"w");
  a=ndioShape(src);
  b=ndref(a,malloc(ndnbytes(a)),nd_heap);
  ndioClose(ndioRead(src,b));
  if(t!=nd_id_unknown)
    b=ndconvert_ip(b,t);
  ndioClose(ndioWrite(dst,b));
Finalize:
  ndfree(a);
  return isok;
Error:
  isok=0;
  goto Finalize;
}

int main(int argc,char* argv[])
{ struct _opt_t opts;
  int isok=0;
  if(!optparse(&opts,argc,argv)) return -1;
  isok=doone(opts.dstname,opts.srcname,opts.type);
Finalize:  
  return !isok;
}
