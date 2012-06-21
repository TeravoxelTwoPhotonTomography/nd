#include "nd.h"
#include "src/io/interface.h"
#include "image.h"
#include <string.h>

#define ENDL              "\n"
#define LOG(...)          fprintf(stderr,__VA_ARGS__)
#define TRY(e)            do{if(!(e)) { LOG("%s(%d):"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,#e); goto Error;}} while(0)
#define NEW(type,e,nelem) TRY((e)=(type*)malloc(sizeof(type)*(nelem)))
#define SAFEFREE(e)       if(e){free(e); (e)=NULL;}
#define FAIL              do{ LOG("Execution should not have reached this point."ENDL); goto Error; }while(0)

// Type Translation Tables
static const
nd_type_id_t types_mylib_to_nd[] =
  {
    nd_u8,
    nd_u16,
    nd_u32,
    nd_u64,
    nd_i8,
    nd_i16,
    nd_i32,
    nd_i64,
    nd_f32,
    nd_f64,
  };

static const
Value_Type types_nd_to_mylib[] =
  {
    UINT8_TYPE,
    UINT16_TYPE,
    UINT32_TYPE,
    UINT64_TYPE,
    INT8_TYPE,
    INT16_TYPE,
    INT32_TYPE,
    INT64_TYPE,
    FLOAT32_TYPE,
    FLOAT64_TYPE
  };

/////
///// INTERFACE
/////

static const char* name_tiff(void) { return "tiff/mylib"; }

static unsigned is_tiff(const char *path, const char *mode)
{ Tiff *t=Open_Tiff((char*)path,(char*)mode);
  if(t) Close_Tiff(t);
  return t!=NULL;
}

static void* open_tiff(const char* path, const char *mode)
{ return Open_Tiff((char*)path,(char*)mode);
}

// Following functions will log to the file object.
#undef  LOG
#define LOG(...) ndioLogError(file,__VA_ARGS__)

static void close_tiff(ndio_t file)
{ Tiff *ctx;
  if(!file) return;
  ctx=(Tiff*)ndioContext(file);
  if(ctx) Close_Tiff(ctx);
}

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

#define countof(e) (sizeof(e)/sizeof(*e))

static nd_t shape_tiff(ndio_t file)
{ 
  int w,h,c,d;
  Tiff *ctx;
  TRY(ctx=(Tiff*)ndioContext(file));
  // assumes first ifd is representative
  Rewind_Tiff(ctx); // just in case? I don't know if there's a way to remember the ifd we we're on 
  TRY(0==Get_IFD_Shape(ctx,&w,&h,&c));
  for(d=0;!Tiff_EOF(ctx);Advance_Tiff(ctx),++d); // count planes
  Rewind_Tiff(ctx);
  { nd_t out=ndinit();
    size_t k,shape[]={w,h,d,c};
    k=pack(shape,countof(shape));    
    ndref(out,NULL,prod(shape,k));
    ndcast(out,types_mylib_to_nd[Get_IFD_Channel_Type(ctx,0)]);
    ndreshape(out,(unsigned)k,shape);
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

/** Assumes:
    1. All planes have the same size
    2. Output ordering is w,h,d,c
    3. All channels must have the same type
    3. Array container has the correct size and type
*/
static unsigned read_tiff(ndio_t file, nd_t a)
{ size_t i;
  int w,h,ichan,nchan,isok=1;
  void  *d;
  Tiff *ctx;
  Array *plane=0;
  
  TRY(ctx=(Tiff*)ndioContext(file));           /// \todo these checks should be done by the higher level interface, and the documentation should reflect that these pointers are gauranteed not null.
  TRY(ndkind(a)==nd_cpu && (d=nddata(a))!=NULL);

  TRY(ndndim(a)>=2);

  TRY(0==Get_IFD_Shape(ctx,&w,&h,&nchan));
  { Dimn_Type dims[2] = {w,h};
    TRY(plane=Make_Array_Of_Data(
        PLAIN_KIND,
        Get_IFD_Channel_Type(ctx,0),
        2,
        dims,
        nddata(a)));
  }
  { const size_t chanstride = (nchan>1)?ndstrides(a)[3]:0;
    for(ichan=0;ichan<nchan;++ichan)
    { for(i=0,Rewind_Tiff(ctx);
          !Tiff_EOF(ctx);
          ++i,Advance_Tiff(ctx)) // strides: (1,w,wh,[whd,whdc])
      { plane->data=(void*)((uint8_t*)nddata(a)+i*ndstrides(a)[2]+ichan*chanstride);
        TRY(0==Get_IFD_Channel(ctx,ichan,plane));
      }
      Rewind_Tiff(ctx);
    }
  }
Finalize:
  if(plane) {plane->data=0; Free_Array(plane);}
  return isok;
Error:
  isok=0;
  goto Finalize;
}

static unsigned write_tiff(ndio_t file, nd_t a)
{ size_t w,h,d,c;
  Tiff  *ctx;
  Array *plane=0;
  int    is_ok=1;
  TRY(ctx=(Tiff*)ndioContext(file));           /// \todo these checks should be done by the higher level interface, and the documentation should reflect that these pointers are gauranteed not null.
  TRY(ndkind(a)==nd_cpu && nddata(a)!=NULL);
  switch(ndndim(a))
  { case 0:
    case 1: TRY(ndndim(a)>1); //reject 0d or 1d
    case 2: c=d=1;                  break;
    case 3: c=1;   d=ndshape(a)[2]; break;
    default:
      c=ndshape(a)[ndndim(a)-1];                   // channels are the last dimension
      d=ndstrides(a)[ndndim(a)-1]/ndstrides(a)[2]; // concatenate dimensions 3...ndim-1, inclusive //1,w,wh,whd,whdt,whdtc
  }
  w=ndshape(a)[0];
  h=ndshape(a)[1];
  
  { Dimn_Type dims[2] = {(Dimn_Type)w,(Dimn_Type)h};
    TRY(plane=Make_Array_Of_Data(
        PLAIN_KIND,
        types_nd_to_mylib[ndtype(a)],
        2,dims,
        nddata(a)));
  }
  { size_t i,j;
    const size_t chanstride = (c>1)?ndstrides(a)[3]:0;
    for(i=0;i<d;++i)
    { for(j=0;j<c;++j)
      { Channel_Kind k= (Channel_Kind)((c<3)?(RED_CHAN+j):PLAIN_CHAN);
        plane->data=(void*)((uint8_t*)nddata(a)+i*ndstrides(a)[2]+j*chanstride);        
        TRY(0==Add_IFD_Channel(ctx,plane,k));
      }
      Update_Tiff(ctx,LZW_PRESS);
    }
  }
  
Finalize:
  if(plane) {plane->data=NULL; Free_Array(plane);}
  return is_ok;
Error:
  is_ok=0;
  goto Finalize;
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
  api.name   = name_tiff;
  api.is_fmt = is_tiff;
  api.open   = open_tiff;
  api.close  = close_tiff;
  api.shape  = shape_tiff;
  api.read   = read_tiff;
  api.write  = write_tiff;
  return &api;
}
