/** \file
    FFMPEG Interface for nD IO.

    \section ndio-ffmpeg-read

    Two options for data flow:
    1. Try to decode directly into destination array
    2. Decode frame into temporary buffer and then copy to destination array.

    Option (2) is simpler to code.  I think you have to override some
    (get|release)_buffer functions somewhere in order to do option 1.  Another
    advantage of option 2 is that we get to choose how to translate strange
    pixel formats.

    \author Nathan Clack
    \date   June 2012

    \todo writer
    \todo handle options
    \todo channels as multiple video streams?
    \todo readable pixel formats?
          see pixdesc.[hc] and av_pix_fmt_descriptors[PIX_FMT_NB]
                               av_(write|read)_image_line
*/
#include "nd.h"
#include "src/io/interface.h"
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

#define ENDL              "\n"
#define LOG(...)          fprintf(stderr,__VA_ARGS__)
#define TRY(e)            do{if(!(e)) { LOG("%s(%d):"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,#e); goto Error;}} while(0)
#define NEW(type,e,nelem) TRY((e)=(type*)malloc(sizeof(type)*(nelem)))
#define SAFEFREE(e)       if(e){free(e); (e)=NULL;}
#define FAIL              do{ LOG("Execution should not have reached this point."ENDL); goto Error; }while(0)

#define AVTRY(expr,msg) \
  do{                                                       \
    int v=(expr);                                           \
    if(v<0 && v!=AVERROR_EOF)                               \
    { char buf[1024];                                       \
      av_strerror(v,buf,sizeof(buf));                       \
      LOG("%s(%d):"ENDL "%s"ENDL "%s"ENDL "FFMPEG: %s"ENDL, \
          __FILE__,__LINE__,#expr,(char*)msg,buf);          \
      goto Error;                                           \
    }                                                       \
  }while(0)

static int is_one_time_inited = 0; /// \todo should be mutexed

struct _ndio_ffmpeg_t
{ AVFormatContext   *fmt;    ///< The main handle to the open file
  struct SwsContext *sws;    ///< software scaling context
  int                ivideo; ///< stream index
} *ndio_ffmpeg_t;

/////
///// HELPERS
/////

#define CCTX(e) ((e)->ctx->streams[(e)->ivideo]->codec) ///< gets the AVCodecContext for the selected video stream

/** One-time initialization for ffmpeg library.

    This gets called by ndio_get_format_api(), so it's guaranteed to be called
    before any of the interface implementation functions.
 */
static void maybe_init()
{ if(is_one_time_inited)
    return;
  avcodec_register_all();
  av_register_all();
  avformat_network_init();
  is_one_time_inited = 1;
}

#include <libavutil/pixdesc.h>
int pixfmt_to_nd_type(int pxfmt, nd_type_id_t *type, int *nchan)
{ int ncomponents   =av_pix_fmt_descriptors[pxfmt].ncomponents;
  int bits_per_pixel=av_get_bits_per_pixel(av_pix_fmt_descriptors+pxfmt);
  TRY(ncomponents>0);
  int bytes=(int)ceil(bits_per_pixel/ncomponents/8.0f);
  *nchan=ncomponents;
  switch(nbytes)
  { case 1: *type=u8;  return 1;
    case 2: *type=u16; return 1;
    case 4: *type=u32; return 1;
    case 8: *type=u64; return 1;
    default:
      ;
  }
Error:
  return 0;
}

PIX_FMT pixfmt_to_output_pixfmt(int pxfmt)
{ int ncomponents   =av_pix_fmt_descriptors[pxfmt].ncomponents;
  int bits_per_pixel=av_get_bits_per_pixel(av_pix_fmt_descriptors+pxfmt);
  int bytes=ncomponents?(int)ceil(bits_per_pixel/ncomponents/8.0f):0;
  return to_pixfmt(bytes,ncomponents);
}

PIX_FMT to_pixfmt(int nbytes, int ncomponents)
{
  switch(ncomponents)
  { case 1:
      switch(nbytes)
      { case 1: return PIX_FMT_GRAY8;
        case 2: return PIX_FMT_GRAY16;
        default:;
      }
      break;
    case 2: // for write
    case 3:
      switch(nbytes)
      { case 1: return PIX_FMT_RGB24;
        case 2: return PIX_FMT_RGB48;
        default:;
      }
      break;
    case 4:
      switch(nbytes)
      { case 1: return PIX_FMT_RGBA32;
        case 2: return PIX_FMT_RGBA64;
        default:;
      }
      break;
    default:
      ;
  }
Error:
  return PIX_FMT_NONE;
}

/////
///// INTERFACE
/////

static const char* name_ffmpeg(void) { return "ffmpeg"; }

static unsigned is_ffmpeg(const char *path, const char *mode)
{ AVFormatContext *fmt=0;
  if(mode[0]!='r') return 0; // can only read for now
  // just check that container can be opened; don't worry about streams, etc...
  if(avformat_open_input(&fmt,path,NULL/*input format*/,NULL/*options*/),path)
  { av_close_input_file(fmt);
    return 1;
  }
  return 0;
}


static void* open_ffmpeg(const char* path, const char *mode)
{ ndio_ffmpeg_t *self=0;
  TRY(mode[0]=='r'); // only accept read for right now, write not implemented yet
  NEW(struct _ndio_ffmpeg_t,self,1);
  memset(self,0,sizeof(*self));

  AVTRY(avformat_open_input(&self->fmt,path,NULL/*input format*/,NULL/*options*/),path);
  AVTRY(avformat_find_stream_info(self->fmt,NULL),"Failed to find stream information.");
  { AVCodec *codec=0;
    AVTRY(self->ivideo=av_find_best_stream(self->fmt,AVMEDIA_TYPE_VIDEO,-1,-1,&codec,0/*flags*/),"Failed to find a video stream.");
    AVTRY(avcodec_open2(CCTX(self),codec,NULL/*options*/),"Cannot open video decoder."); // inits the selected stream's codec context

    TRY(self->sws=sws_getContext(codec->width,codec->height,codec->pix_fmt,
                                 codec->width,codec->height,pixfmt_to_output_pixfmt(codec->pix_fmt)));
  }

  return self;
Error:
  if(self) free(self);
  return NULL;
}

// Following functions will log to the file object.
#undef  LOG
#define LOG(...) ndioLogError(file,__VA_ARGS__)

static void close_ffmpeg(ndio_t file)
{ ndio_ffmpeg_t *self;
  if(!file) return;
  if(!(self=(ndio_ffmpeg_t)ndioContext(file)) ) return;
  if(CCTX(self)) avcodec_close(CCTX(self));
  if(self->sws)  sws_freeContext(self->sws);
  if(self->ctx)  av_close_input_file(self->ctx);
  free(self);
}

#define countof(e) (sizeof(e)/sizeof(*e))


static int pack(size_t *s, int n)
{ int i,c;
  for(i=0,c=0;i<n;++i)
  { s[c]=s[i];
    if(s[i]!=1) ++c;
  }
  return c;
}

static size_t prod(const size_t *s, size_t n)
{ size_t i,p=1;
  for(i=0;i<n;++i) p*=s[i];
  return p;
}

static nd_t shape_ffmpeg(ndio_t file)
{ int w,h,d,c;
  nd_type_id_t type;
  ndio_ffmpeg_t *self;
  AVCodecContext *codec;
  TRY(file);
  TRY(self=(ndio_ffmpeg_t)ndioContext(file));
  TRY(codec=CCTX(self));
#if 0  // was in the loader for the whisker tracking code.  Don't remember why.
  /* Frame rate fix for some codecs */
  if( ret->pCtx->time_base.num > 1000 && ret->pCtx->time_base.den == 1 )
    ret->pCtx->time_base.den = 1000;
#endif
  d=(int)((self->ctx->duration/(double)AV_TIME_BASE)*codec->time_base.den);
  w=codec->width;
  h=codec->height;
  TRY(pixfmt_to_nd_type(codec->pix_fmt,&type,&c));
  { nd_t out=ndinit();
    size_t k,shape[]={w,h,d,c};
    k=pack(shape,countof(shape));
    ndref(out,NULL,prod(shape,k));
    ndcast(out,type);
    ndreshape(out,(unsigned)k,shape);
    return out;
  }
Error:
  return NULL;
}

/** Assumes:
    1. Output ordering is w,h,d,c
    2. All channels must have the same type
    3. Array container has the correct size and type
*/
static unsigned read_ffmpeg(ndio_t file, nd_t a)
{
}

static unsigned write_ffmpeg(ndio_t file, nd_t a)
{
}

/////
///// EXPORT
/////

#ifdef _MSC_VER
#define shared __declspec(dllexport)
#else
#define shared
#endif

shared const ndio_fmt_t* ndio_get_format_api(void)
{ static ndio_fmt_t api = {0};
  maybe_init();
  api.name   = name_ffmpeg;
  api.is_fmt = is_ffmpeg;
  api.open   = open_ffmpeg;
  api.close  = close_ffmpeg;
  api.shape  = shape_ffmpeg;
  api.read   = read_ffmpeg;
  api.write  = write_ffmpeg;
  return &api;
}

