/**
 * \file
 * An ndio plugin for reading/writing hdf5 files.
 *
 * HDF5 seems to support a dizzying array of features.  This plugin won't
 * support them all.  Instead, the focus will be on implementing core support
 * for reading and writing dense arrays with an emphasis on the kinds of files
 * that matlab and python/numpy might produce.
 *
 * Probably the two most important hdf5 features not supported are:
 * 1. Hierarchical organization.
 * 2. Sparsely writting chunks of data.
 *
 * Only the first data set encountered is the one that is read.
 *
 * Chunking is set up to support appending arrays to a data set.
 */

#include "hdf5"

/// Recognized file extensions (A NULL terminated list).
static const char *g_exts[]={
  '.h5','.mat','.hdf','.hdf4',
  '.h4','.he5','.he4','.hdf5',
  NULL};

/// @cond DEFINES
#define PATHSEP "/"
#define countof(e) (sizeof(e)/sizeof(*e))
#define ENDL                  "\n"
#define LOG(...)              printf(__VA_ARGS__)
#define TRY(e)                do{if(!(e)) { LOG("%s(%d): %s()"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error;}} while(0)
#define HTRY(e)               do{if((e)<0){ LOG("%s(%d): %s()"ENDL "\tHDF5 Expression evaluated to a negative value."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error;}} while(0) 
#define TRYMSG(e,msg)         do{if(!(e)) { LOG("%s(%d): %s()"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL "\t%sENDL",__FILE__,__LINE__,__FUNCTION__,#e,msg); goto Error; }}while(0)
#define FAIL(msg)             do{ LOG("%s(%d): %s()"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,msg); goto Error;} while(0)
#define RESIZE(type,e,nelem)  TRY((e)=(type*)realloc((e),sizeof(type)*(nelem)))
#define NEW(type,e,nelem)     TRY((e)=(type*)malloc(sizeof(type)*(nelem)))
#define SAFEFREE(e)           if(e){free(e); (e)=NULL;}
/// @endcond

typedef struct _ndio_hdf5_t* ndio_hdf5_t;
struct _ndio_hdf5_t
{ hid_t file,
        dataset,
        space,
        type;
  char  isr,
        isw;
  char *name;
};

//
// === HELPERS ===
//

static void init(ndio_hdf5_t self)
{ memset(self,0xff,sizeof(*self)); // set everything to -1
  self->isr =0;
  self->isw =0;
  self->name=0;
}

static void close(ndio_hdf5_t self)
{ if(!self) return;
#define RELEASE(f,e) if(self->e>=0) {f(self->e); self->e=-1;}
  RELEASE(H5Fclose,file);
  RELEASE(H5Dclose,dataset);
  RELEASE(H5Sclose,space);
  RELEASE(H5Tclose,type);
#undef RELEASE 
#define RELEASE(f,e) if(self->e) {f(self->e); (self->e)=0;}
  RELEASE(free,name);
#undef RELEASE   
  free(self);
}

static herr_t query_first_name(hid_t id,const char *name,const H5L_info_t *i, void *data)
{ H5O_info_t info;
  HTRY(H5Oget_info_by_name(id,name,&info,H5P_DEFAULT));
  if(info.type==H5O_TYPE_DATASET)
  { TRY(data=malloc(1+strlen(name)));
    strcpy((char*)data,name);
  }
  return 1; // imediately terminate iteration
Error:
  return 0;
}

static char* name(ndio_hdf5_t self)
{ static const char default[]="data";
  if(self->name) return self->name;
  if(!self->isr)
  { TRY(self->name=malloc(1+countof(default)));
    strcpy(self->name,default);
  }
  else
  { HTRY(H5Literate(hid,H5_INDEX_NAME,H5_ITER_NATIVE,NULL,query_first_name,&self->name));
  }
  return self->name;
Error:
  return 0;
}

static hid_t dataset(ndio_hdf5_t self)
{ char *name_=0;
  if(self->dataset>-1) return self->dataset;
  TRY(self->isr);
  HTRY(self->dataset=H5Dopen(self->file,name(self),H5P_DEFAULT)); /*data access property list - includes chunk cache*/
  return self->dataset;
Error:
  return -1;
}

static hid_t make_space(ndio_hdf5_t self,,unsigned ndim,size_t *shape)
{ hsize_t *maxdims=0;
  TRY(self->space==-1); // how to handle when space is already created?
  TRY(self->isw);
  TRY(maxdims=(hsize_t*)alloca(sizeof(*maxdims)*ndim));

  return self->space;
Error:
  return -1;
}

static hid_t make_dataset(ndio_hdf_t self,nd_type_id_t typeid,unsigned ndim,size_t *shape)
{ TRY(self->dataset==-1); // how to handle when dataset is already created?
  TRY(self->isw);
  HTRY(self->dataset=H5Dcreate(
                     self->file,name(self),
                     nd_to_hdf5_type(typeid),/*typeid*/
                     make_space(self,ndims,shape),
                     H5P_DEFAULT,/*(rare)     link    creation props*/
                     H5P_DEFAULT,/*(frequent) dataset creation props*/
                     H5P_DEFAULT /*(rare)     dataset access   props*/
                     ));
  return self->dataset;
Error:
  return -1;
}

/// Type translation: hdf5->nd
static nd_kind_t hdf5_to_nd_type(hid_t tid)
{ size_t        p=H5Tget_precision(tid);
  
  switch(H5Tget_class(tid))
  { case H5T_INTEGER:
      { H5T_sign_t sign=H5Tget_sign(tid);
        switch(p)
        { 
#define   CASE(n) case n:  return (sign==H5T_SGN_NONE)?nd_u##n:nd_i##n
          CASE(8);
          CASE(16);
          CASE(32);
          CASE(64);
#undef    CASE
          default:;
        }
      }
      return nd_id_unknown;
    case H5T_FLOAT:
      if     (p==32) return nd_f32_t;
      else if(p==64) return nd_64_t;
    default:         return nd_id_unknown;
  }
}

static hid_t nd_to_hdf5_type(nd_type_id_t tid)
{ static const hid_t table[]=
  { 
#define T(e) H5T_NATIVE_#e
    T(UCHAR), T(USHORT), T(UINT), T(ULLONG),
    T(CHAR),  T(SHORT),  T(INT),  T(LLONG),
    T(FLOAT), T(DOUBLE)
#undef T
  }
  TRY(nd_id_unknown<tid && tid<nd_id_count);
  return 1;
Error:
  return -1;
}

static unsigned parse_mode_string(const char* mode,char *isr, char *isw)
{ char* c;
  char r,w,ix;
  *isr=*isw=r=w=x=0;
  for(c=(char*)mode;*c;++c)
  { switch(*c)
    { case 'r': r=1; break;
      case 'w': w=1; break;
      case 'x': x=1; break;
      default:
        FAIL;
    }
  }
  *isr=r;
  *isw=w;
  if(r&&w) return H5F_ACC_RDWR;
  if(r)    return H5F_ACC_RDONLY;
  if(w&&x) return H5F_ACC_EXCL;
  if(w)    return H5F_ACC_TRUNC;
Error:
  *isr=1;
  *isr=0;
  return H5F_ACC_RDONLY;
}

//
// === INTERFACE ===
//

/// Format name. \returns "hdf5"
static const char* hdf5_fmt_name(void) {return "hdf5";}

/**
 * Format detection.
 * If the file is opened for read mode, HDF5's internal formast detection
 * will be used.  Otherwise, the function will look for a file extension match.
 */ 
static unsigned hdf5_is_fmt(const char* path, const char* mode)
{ char isr,isw,*e,*ext;  
  parse_mode_string(mode,&isr,&isw);
  if(isr) 
    return H5Fis_hdf5(path);
  e=strrchr(path,'.'); 
  for(ext=g_exts;*ext;++ext)
    if strcmp(e,ext) return 1;
  return 0;
}

/**
 * Open.
 */
static void* hdf5_open(const char* path, const char* mode)
{ ndio_hdf5_t self=0;
  unsigned mode;
  NEW(struct _ndio_hdf5_t,self,1);
  init(self);
  mode=parse_mode_string(mode,&self->isr,&self->isw);
  if(!self->isr && self->isw)
    HTRY(self->hid=H5Fcreate(path,         // filename
                             mode,         // mode flag (unsigned)
                             H5P_DEFAULT,  // creation property list
                             H5P_DEFAULT));// access property list
  else if(self->isr)
    HTRY(self->hid=H5Fopen(path,mode,H5P_DEFAULT)); // uses file access property list
  return self;
Error:
  close(self);
  return 0;
}

/**
 * Close the file and release resources.
 */
static void hdf5_close(ndio_t file)
{ if(!file) return;
  close((ndio_hdf5_t)ndioContext(file));
}

static void hdf5_shape(ndio_t file)
{ ndio_hdf5_t *self=(ndio_hdf5_t)ndioContext(file);

}

//
// === EXPORT ===
//

/// @cond DEFINES
#ifdef _MSC_VER
#define shared extern "C" __declspec(dllexport)
#else
#define shared extern "C"
#endif
/// @endcond

#include "src/io/interface.h"
/// Interface function for the ndio-hdf5 plugin.
shared const ndio_fmt_t* ndio_get_format_api(void)
{ 
  static ndio_fmt_t api=
  { hdf5_fmt_name,
    hdf5_is_fmt,
    hdf5_open,
    hdf5_close,
    hdf5_shape,
    hdf5_read,
    hdf5_write,
    NULL, //set
    NULL, //get
    hdf5_canseek,
    hdf5_seek,
    NULL 
  };
  return &api;
Error:
  return 0;
}