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

    Writer
    -----
    \todo I'm getting the duration or pts or something wrong.
          VLC and browsers don't know how long the videos are.
    \todo map signed ints to unsigned before encoding.  sws scale
          doesn't know signed.
    \todo handle options

    Reader
    ------
    \todo what to do with multiple video streams?
    \todo for appropriate 1d data, use audio streams

    \section ndio-ffmpeg-notes Notes

        - duration be crazy
          - Had this at one point:
            self->nframes  = ((DURATION(self)/(double)AV_TIME_BASE)*cctx->time_base.den);
            Worked for the whisker tracker? But now seems DURATION(self) gives the
            right number of frames.
          - The AVFormatContext.duration/AV_TIME_BASE seems to be the duration in seconds

        - FFMPEG API documentation has lots of examples, but it's hard to know which one's to
          use.  Some of the current examples use depricated APIs.
          - The process of read/writing a container file is called demuxing/muxing.
          - The process of unpacking/packing a video stream is called decoding/encoding.
*/
#include "nd.h"
#include "src/io/interface.h"
#include <stdint.h>
#include <string.h>

// need to define inline before including av* headers on C89 compilers
#ifdef _MSC_VER
#define inline __forceinline
#endif
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libavutil/pixdesc.h"
#include "libavutil/opt.h"
#include "libavutil/imgutils.h"

#define ENDL              "\n"
#define LOG(...)          fprintf(stderr,__VA_ARGS__)
#define TRY(e)            do{if(!(e)) { LOG("%s(%d): %s"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error;}} while(0)
#define NEW(type,e,nelem) TRY((e)=(type*)malloc(sizeof(type)*(nelem)))
#define SAFEFREE(e)       if(e){free(e); (e)=NULL;}
#define FAIL(msg)         do{ LOG("%s(%d): %s"ENDL "\tExecution should not have reached this point."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,msg); goto Error; }while(0)

#define AVTRY(expr,msg) \
  do{                                                       \
    int v=(expr);                                           \
    if(v<0 && v!=AVERROR_EOF)                               \
    { char buf[1024];                                       \
      av_strerror(v,buf,sizeof(buf));                       \
      LOG("%s(%d):"ENDL "\t%s"ENDL "\t%s"ENDL "\tFFMPEG: %s"ENDL, \
          __FILE__,__LINE__,#expr,(char*)msg,buf);          \
      goto Error;                                           \
    }                                                       \
  }while(0)

static int is_one_time_inited = 0; /// \todo should be mutexed

typedef struct _ndio_ffmpeg_t
{ AVFormatContext   *fmt;     ///< The main handle to the open file
  struct SwsContext *sws;     ///< software scaling context
  AVFrame           *raw;     ///< frame buffer for holding data before translating it to the output nd_t format
  int                istream; ///< stream index
  int64_t            nframes; ///< duration of video in frames (for reading)
  AVDictionary      *opts;
} *ndio_ffmpeg_t;

/////
///// HELPERS
/////

#define CCTX(e)     ((e)->fmt->streams[(e)->istream]->codec)    ///< gets the AVCodecContext for the selected video stream
#define DURATION(e) ((e)->fmt->streams[(e)->istream]->duration) ///< gets the duration in some sort of units

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

  av_log_set_level(0
#if 0
    |AV_LOG_DEBUG
    |AV_LOG_VERBOSE
    |AV_LOG_INFO
    |AV_LOG_WARNING
    |AV_LOG_ERROR
    |AV_LOG_FATAL
    |AV_LOG_PANIC
#endif
    );
}

int pixfmt_to_nd_type(int pxfmt, nd_type_id_t *type, int *nchan)
{ int ncomponents   =av_pix_fmt_descriptors[pxfmt].nb_components;
  int bits_per_pixel=av_get_bits_per_pixel(av_pix_fmt_descriptors+pxfmt);
  int nbytes;
  TRY(ncomponents>0);
  nbytes=(int)ceil(bits_per_pixel/ncomponents/8.0f);
  *nchan=ncomponents;
  switch(nbytes)
  { case 1: *type=nd_u8;  return 1;
    case 2: *type=nd_u16; return 1;
    case 4: *type=nd_u32; return 1;
    case 8: *type=nd_u64; return 1;
    default:
      ;
  }
Error:
  return 0;
}

int to_pixfmt(int nbytes, int ncomponents)
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
      { case 1: return PIX_FMT_RGBA;
        case 2: return PIX_FMT_RGBA64;
        default:;
      }
      break;
    default:
      ;
  }
  return PIX_FMT_NONE;
}

int pixfmt_to_output_pixfmt(int pxfmt)
{ int ncomponents   =av_pix_fmt_descriptors[pxfmt].nb_components;
  int bits_per_pixel=av_get_bits_per_pixel(av_pix_fmt_descriptors+pxfmt);
  int bytes=ncomponents?(int)ceil(bits_per_pixel/ncomponents/8.0f):0;
  return to_pixfmt(bytes,ncomponents);
}


/////
///// INTERFACE
/////

static const char* name_ffmpeg(void) { return "ffmpeg"; }

static unsigned test_readable(const char *path)
{ AVFormatContext *fmt=0;
  // just check that container can be opened; don't worry about streams, etc...
  if(0==avformat_open_input(&fmt,path,NULL/*input format*/,NULL/*options*/))
  {
    int ok=(0<=avformat_find_stream_info(fmt,NULL));
    if(ok) //check the codec
    { AVCodec *codec=0;
      AVCodecContext *cctx=0;
      int i=av_find_best_stream(fmt,AVMEDIA_TYPE_VIDEO,-1,-1,&codec,0/*flags*/);
      if(!codec || codec->id==CODEC_ID_TIFF) //exclude tiffs because ffmpeg can't properly do multiplane tiff
        ok=0;
      //cctx=fmt->streams[i]->codec;
      //if(cctx) avcodec_close(cctx);
    }
    avformat_close_input(&fmt);
    return ok;
  }
  return 0;
}

static unsigned test_writable(const char *path)
{ AVOutputFormat *fmt=0;
  const char *ext;
  ext=(ext=strrchr(path,'.'))?(ext+1):""; // yields something like "mp4" or, if no extension found, "".
  while( fmt=av_oformat_next(fmt) )
  { if(0==strcmp(fmt->name,ext))
      return fmt->video_codec!=CODEC_ID_NONE;
  }
  return 0;
}

static unsigned is_ffmpeg(const char *path, const char *mode)
{
  switch(mode[0])
  { case 'r': return test_readable(path);
    case 'w': return test_writable(path);
    default:
      ;
  }
  return 0;
}

static ndio_ffmpeg_t open_reader(const char* path)
{ ndio_ffmpeg_t self=0;
  NEW(struct _ndio_ffmpeg_t,self,1);
  memset(self,0,sizeof(*self));

  TRY(self->raw=avcodec_alloc_frame());
  AVTRY(avformat_open_input(&self->fmt,path,NULL/*input format*/,NULL/*options*/),path);
  AVTRY(avformat_find_stream_info(self->fmt,NULL),"Failed to find stream information.");
  { AVCodec        *codec=0;
    AVCodecContext *cctx=CCTX(self);
    AVTRY(self->istream=av_find_best_stream(self->fmt,AVMEDIA_TYPE_VIDEO,-1,-1,&codec,0/*flags*/),"Failed to find a video stream.");
    AVTRY(avcodec_open2(cctx,codec,NULL/*options*/),"Cannot open video decoder."); // inits the selected stream's codec context
    TRY(self->sws=sws_getContext(cctx->width,cctx->height,cctx->pix_fmt,
                                 cctx->width,cctx->height,pixfmt_to_output_pixfmt(cctx->pix_fmt),
                                 SWS_BICUBIC,NULL,NULL,NULL));

    self->nframes  = DURATION(self);
  }

  return self;
Error:
  if(self)
  { if(CCTX(self)) avcodec_close(CCTX(self));
    if(self->opts) av_dict_free(&self->opts);
    if(self->raw)  av_free(self->raw);
    if(self->sws)  sws_freeContext(self->sws);
    if(self->fmt)  avformat_free_context(self->fmt);
    free(self);
  }
  return NULL;
}

static ndio_ffmpeg_t open_writer(const char* path)
{ ndio_ffmpeg_t self=0;
  AVCodec *codec;
  NEW(struct _ndio_ffmpeg_t,self,1);
  memset(self,0,sizeof(*self));
  TRY(self->raw=avcodec_alloc_frame());

  AVTRY(avformat_alloc_output_context2(&self->fmt,NULL,NULL,path), "Failed to detect output file format from the file name.");
  TRY(self->fmt->oformat && self->fmt->oformat->video_codec!=CODEC_ID_NONE); //Assert that this is a video output format
  TRY(codec=avcodec_find_encoder(self->fmt->oformat->video_codec));
  TRY(avformat_new_stream(self->fmt,codec)); //returns the stream, assume it's index 0 from here on out (self->istream=0 is accurate)
  // maybe open the output file
  TRY(0==self->fmt->flags&AVFMT_NOFILE);                              //if the flag is set, don't need to open the file (I think).  Assert here so I get notified of when this happens.  Expected to be rare/never.
  AVTRY(avio_open(&self->fmt->pb,path,AVIO_FLAG_WRITE),"Failed to open output file.");

  CCTX(self)->codec=codec; // setting this here so we can remember it later.
  AVTRY(CCTX(self)->pix_fmt=codec->pix_fmts[0],"Codec indicates that no pixel formats are supported.");
  return self;
Error:
  if(self->fmt->pb) avio_close(self->fmt->pb);
  if(self->opts) av_dict_free(&self->opts);
  if(self)
  { if(self->fmt)  avformat_free_context(self->fmt);
    free(self);
  }
  return NULL;
}

static int maybe_init_codec_ctx(ndio_ffmpeg_t self, int width, int height, int fps, int src_pixfmt)
{ AVCodecContext *cctx=CCTX(self);
  AVCodec *codec=cctx->codec;
  if(!cctx->width)
  { cctx->width=width;
    cctx->height=height;
    cctx->time_base.num=1;
    cctx->time_base.den=fps;
    cctx->gop_size=12;
    AVTRY(avcodec_open2(cctx,codec,&self->opts),"Failed to initialize encoder.");

    TRY(self->sws=sws_getContext(
      width,height,src_pixfmt,
      width,height,cctx->pix_fmt,
      SWS_BICUBIC,NULL,NULL,NULL));

    TRY(av_image_alloc(self->raw->data,self->raw->linesize,width,height,cctx->pix_fmt,1));
    AVTRY(avformat_write_header(self->fmt,&self->opts),"Failed to write header.");
  }
  return 1;
Error:
  return 0;
}

static void* open_ffmpeg(const char* path, const char *mode)
{
  switch(mode[0])
  { case 'r': return open_reader(path);
    case 'w': return open_writer(path);
    default:
      FAIL("Could not recognize mode.");
  }
Error:
  return 0;
}

// Following functions will log to the file object.
#undef  LOG
#define LOG(...) ndioLogError(file,__VA_ARGS__)

static void close_ffmpeg(ndio_t file)
{ ndio_ffmpeg_t self;
  if(!file) return;
  if(!(self=(ndio_ffmpeg_t)ndioContext(file)) ) return;
  if(CCTX(self))    avcodec_close(CCTX(self));
  if(self->opts)    av_dict_free(&self->opts);
  if(self->raw)     av_free(self->raw);
  if(self->sws)     sws_freeContext(self->sws);
  if(self->fmt)
  { if(self->fmt->oformat)
    AVTRY(av_write_trailer(self->fmt),"Failed to write trailer.");
Error: //ignore error
    if(self->fmt->pb) avio_close(self->fmt->pb);
    avformat_free_context(self->fmt);
  }
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

/**
 * \todo FIXME: Must decode first frame to ensure codec context has the proper width.
 * \todo FIXME: side effect, seeks to begining of video.  Should leave current seek
 *              point unmodified.
 */
static nd_t shape_ffmpeg(ndio_t file)
{ int w,h,d,c;
  nd_type_id_t type;
  ndio_ffmpeg_t self;
  AVCodecContext *cctx;
  AVPacket packet={0};
  TRY(file);
  TRY(self=(ndio_ffmpeg_t)ndioContext(file));
  TRY(cctx=CCTX(self));
  // read first frame to update dimensions from first packet if necessary
  { int fin=0;
    do{
      av_free_packet(&packet);
      AVTRY(av_read_frame(self->fmt,&packet),"Failed to read frame.");
    } while(packet.stream_index!=self->istream);
    AVTRY(avcodec_decode_video2(cctx,self->raw,&fin,&packet),"Failed to decode frame.");
    av_free_packet(&packet);
    AVTRY(av_seek_frame(self->fmt,self->istream,0,AVSEEK_FLAG_BACKWARD/*flags*/),"Failed to seek to beginning.");
  }
#if 0  // was in the loader for the whisker tracking code.  Don't remember why.
  /* Frame rate fix for some codecs */
  if( ret->pCtx->time_base.num > 1000 && ret->pCtx->time_base.den == 1 )
    ret->pCtx->time_base.den = 1000;
#endif
  d=(int)self->nframes;
  w=cctx->width;
  h=cctx->height;
  TRY(pixfmt_to_nd_type(cctx->pix_fmt,&type,&c));
  { nd_t out=ndinit();
    size_t k,shape[]={w,h,d,c};
    k=pack(shape,countof(shape));
    ndref(out,NULL,prod(shape,k));
    ndcast(out,type);
    ndreshape(out,(unsigned)k,shape);
    return out;
  }
Error:
  av_free_packet(&packet);
  return NULL;
}

#if 0
#define DEBUG_PRINT_PACKET_INFO \
    printf("Packet - pts:%5d dts:%5d (%5d) - flag: %1d - finished: %3d - Frame pts:%5d %5d\n",   \
        (int)packet.pts,(int)packet.dts,iframe,                                                  \
        packet.flags,finished,                                                                   \
        (int)self->raw->pts,(int)self->raw->best_effort_timestamp)
#else
#define DEBUG_PRINT_PACKET_INFO
#endif

static void zero(AVFrame *p)
{ int i,j;
  for(j=0;j<4;++j)
    if(p->data[j])
      for(i=0;i<p->height;++i)
        memset(p->data[j]+p->linesize[j]*i,0,p->linesize[j]);
}

/** Parse next packet from current video.
    Advances to the next frame.

    Caller is responsible for passing the correct, pre-allocated plane.

    \returns 1 on success, 0 otherwise.
 */
static int next(ndio_t file,nd_t plane,int iframe)
{ ndio_ffmpeg_t self;
  AVPacket packet = {0};
  int finished = 0;
  TRY(self=(ndio_ffmpeg_t)ndioContext(file));
  do
  { finished=0;
    av_free_packet( &packet ); // no op when packet is null
    AVTRY(av_read_frame(self->fmt,&packet),"Failed to read frame.");   // !!NOTE: see docs on packet.convergence_duration for proper seeking
    if(packet.stream_index!=self->istream)
      continue;
    AVTRY(avcodec_decode_video2(CCTX(self),self->raw,&finished,&packet),NULL);
    // Handle odd cases and debug
    if(CCTX(self)->codec_id==CODEC_ID_RAWVIDEO && !finished)
    { zero(self->raw); // Emit a blank frame.  Something off about the stream.
      finished=1;
    }
    DEBUG_PRINT_PACKET_INFO;
    if(!finished)
      TRY(packet.pts!=AV_NOPTS_VALUE);  // ?don't know what to do when codecs don't provide a pts
  } while(!finished || self->raw->best_effort_timestamp<iframe);
  av_free_packet(&packet);

  /*  === Copy out data, translating to desired pixel format ===
      Assume colors are last dimension.
      Assume plane points to start of image for first color.
      Assume at most four color planes.
      Assume each color plane has identical stride.
      Plane has full dimensionality of parent array; just offset.
  */
  { uint8_t *planes[4]={0};
    int lines[4]={0};
    const int lst = (int) ndstrides(plane)[1],
              cst = (int) ndstrides(plane)[ndndim(plane)-1];
    int i;
    for(i=0;i<countof(planes);++i)
    { lines[i]=lst;
      planes[i]=(uint8_t*)nddata(plane)+cst*i;
    }
    sws_scale(self->sws,              // sws context
              (const uint8_t*const*)self->raw->data, // src slice
              self->raw->linesize,    // src stride
              0,                      // src slice origin y
              CCTX(self)->height,     // src slice height
              planes,                 // dst
              lines);                 // dst line stride
  }

  return 1;
Error:
  av_free_packet( &packet );
  return 0;
}

/** \returns current frame on success, otherwise -1 */
static int seek(ndio_t file, int64_t iframe)
{ ndio_ffmpeg_t self;
  int64_t duration,ts;
  TRY(self=(ndio_ffmpeg_t)ndioContext(file));
  duration = DURATION(self);
  ts = iframe; //av_rescale(duration,iframe,self->nframes);

  TRY(iframe>=0 && iframe<self->nframes);
  AVTRY(avformat_seek_file( self->fmt,       //format context
                            self->istream,   //stream id
                            0,ts,ts,          //min,target,max timestamps
                            0),//AVSEEK_FLAG_ANY),//flags
                            "Failed to seek.");
  avcodec_flush_buffers(CCTX(self));
  //TRY(next(self,iframe));
  return iframe;
Error:
  return -1;
}

static int64_t nframes(const ndio_t file)
{ ndio_ffmpeg_t self;
  TRY(self=(ndio_ffmpeg_t)ndioContext(file));
  return self->nframes;
Error:
  return 0;
}

/** Assumes:
    1. Output ordering is c,w,h,d
    2. Array container has the correct size and type
*/
static unsigned read_ffmpeg(ndio_t file, nd_t a)
{ int64_t i;
  void *o=nddata(a);
  //seek(file,0);
  for(i=0;i<nframes(file);++i,ndoffset(a,2,1))
    TRY(next(file,a,i));
  ndref(a,o,ndnelem(a));
  return 1;
Error:
  return 0;
}

/** Assumes:
    1. Input ordering is c,w,h,d
    2. Appends d planes of a to the stream
 */
static unsigned write_ffmpeg(ndio_t file, nd_t a)
{ ndio_ffmpeg_t self;
  int c,w,h,d,i;
  const size_t *s;
  size_t planestride,colorstride,linestride;
  int pixfmt;
  AVPacket p={0};
  int done;
  AVCodecContext *cctx;

  TRY(self=(ndio_ffmpeg_t)ndioContext(file));
  cctx=CCTX(self);
  s=ndshape(a);
  switch(ndndim(a))
  { case 2:      c=1;      w=s[0];      h=s[1];      d=1;      break;// w,h
    case 3:      c=1;      w=s[0];      h=s[1];      d=s[2];   break;// w,h,d
    case 4:      c=s[0];   w=s[1];      h=s[2];      d=s[3];   break;// c,w,h,d
    default:
      FAIL("Unsupported number of dimensions.");
  }
  planestride=ndstrides(a)[ndndim(a)-1];
  linestride=ndstrides(a)[ndndim(a)-2];
  colorstride=ndstrides(a)[0];
  TRY(PIX_FMT_NONE!=(pixfmt=to_pixfmt(colorstride,c)));
  TRY(maybe_init_codec_ctx(self,w,h,24,pixfmt));
  i=0; done=0;
  while(i<d || !done) // this will push d planes, then repeat the last plane till done.
  { AVFrame *in;
    av_init_packet(&p); // FIXME: for efficiency, probably want to preallocate packet
    if(i<d)
    { const uint8_t* slice[4]={
        ((uint8_t*)nddata(a))+planestride*i+colorstride*0,
        ((uint8_t*)nddata(a))+planestride*i+colorstride*1,
        ((uint8_t*)nddata(a))+planestride*i+colorstride*2,
        ((uint8_t*)nddata(a))+planestride*i+colorstride*3};
      const int stride[4]={linestride,linestride,linestride,linestride};
      in=self->raw;
      sws_scale(self->sws,slice,stride,0,h,self->raw->data,self->raw->linesize);
      self->raw->pts=self->fmt->duration+i;
    } else if(CCTX(self)->codec->capabilities & CODEC_CAP_DELAY) // FIXME: might want to move the close function to properly append data
    { in=NULL;
    }
    AVTRY(avcodec_encode_video2(CCTX(self),&p,in,&done),"Failed to encode packet.");
    if(done)
    { AVTRY(av_write_frame(self->fmt,&p),"Failed to write frame.");
      av_destruct_packet(&p);
    }
    if(i<d) ++i;
  }

  return 1;
Error:
  return 0;
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

