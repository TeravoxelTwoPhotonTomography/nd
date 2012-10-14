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
 *
 * Notes:
 * 1. Dimension ordering in HDF5 is reverse of the convention I use.
 *    My dim[0] is the fastest, theirs is the slowest.
 *    I didn't account for this on first writing.  There may still be some
 *    bugs in here where I failed to reverse things correctly. (Need more
 *    comprehensive tests!)
 */
#pragma warning(disable:4996)
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "hdf5.h"
#include "nd.h"

#include "macros.h" // TRY, LOG, NEW, etc...

#ifdef _MSC_VER
#define alloca _alloca
#endif

/// Recognized file extensions (A NULL terminated list).
static const char *g_readable_exts[]={
  ".h5",".mat",".hdf",".hdf4",
  ".h4",".he5",".he4",".hdf5",
  NULL};
static const char *g_writeable_exts[]={
  ".h5",".hdf",".he5",".hdf5",
  NULL};


typedef struct _ndio_hdf5_t* ndio_hdf5_t;
struct _ndio_hdf5_t
{ hid_t file,
        dataset,
        space,
        type;
  hid_t dataset_creation_properties;
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
  RELEASE(H5Pclose,dataset_creation_properties);
#undef RELEASE 
#define RELEASE(f,e) if(self->e) {f(self->e); (self->e)=0;}
  RELEASE(free,name);
#undef RELEASE   
  free(self);
}

/**
 * \param[in,out]   data    pointer to a a string pointer.
 */
static herr_t query_first_name(hid_t id,const char *name,const H5L_info_t *i, void *data)
{ H5O_info_t info;
  HTRY(H5Oget_info_by_name(id,name,&info,H5P_DEFAULT));
  if(info.type==H5O_TYPE_DATASET)
  { TRY(*(char**)data=malloc(1+strlen(name)));
    strcpy(*(char**)data,name);
  }
  return 1; // imediately terminate iteration
Error:
  return 0;
}

static void copy_hsz_sz(size_t n, hsize_t *dst, size_t *src)
{ size_t i;
  for(i=0;i<n;++i) dst[i]=(hsize_t)src[i];
}

static void reverse_hsz_sz(size_t n, hsize_t *dst, size_t *src)
{ size_t i;
  for(i=0;i<n;++i) dst[n-1-i]=(hsize_t)src[i];
}

/// Type translation: hdf5->nd
static nd_type_id_t hdf5_to_nd_type(hid_t tid)
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
      if     (p==32) return nd_f32;
      else if(p==64) return nd_f64;
    default:         return nd_id_unknown;
  }
}

static hid_t nd_to_hdf5_type(nd_type_id_t tid)
{ /* 
     Cannot just make a static look up table since HDF5 native types are not
     compile time constants.
   */
  switch(tid)
  {
    #define CASE(id,name) case id: return H5T_NATIVE_##name
    CASE(nd_u8 ,UCHAR); CASE(nd_u16,USHORT); CASE(nd_u32,UINT); CASE(nd_u64,ULLONG);
    CASE(nd_i8 ,CHAR);  CASE(nd_i16,SHORT);  CASE(nd_i32,INT);  CASE(nd_i64,LLONG);
    CASE(nd_f32,FLOAT); CASE(nd_f64,DOUBLE);
    default: return -1;
    #undef CASE
  }
}


static char* name(ndio_hdf5_t self)
{ static const char name_[]="data";
  if(self->name) return self->name;
  if(!self->isr)
  { TRY(self->name=malloc(1+countof(name_)));
    strcpy(self->name,name_);
  }
  else
  { HTRY(H5Literate(self->file,H5_INDEX_NAME,H5_ITER_NATIVE,NULL,query_first_name,&self->name));
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

static hid_t space(ndio_hdf5_t self)
{ if(self->space>-1) return self->space;
  return (self->space=H5Dget_space(dataset(self)));
}

static hid_t make_space(ndio_hdf5_t self,unsigned ndims,size_t *shape)
{ hsize_t *maxdims=0,*dims=0;
  TRY(self->space==-1); // how to handle when space is already created?
  TRY(self->isw);
  STACK_ALLOC(hsize_t,maxdims,ndims);
  STACK_ALLOC(hsize_t,dims,ndims);
  { unsigned i;
    for(i=0;i<ndims;++i)
    { maxdims[i]=H5S_UNLIMITED;
      dims[ndims-1-i]=shape[i];
    }
  }
  return self->space=H5Screate_simple(ndims,dims,maxdims);
Error:
  return -1;
}

static hid_t dtype(ndio_hdf5_t self)
{ if(self->type>-1) return self->type;
  return (self->type=H5Dget_type(dataset(self)));
}

static hid_t dataset_creation_properties(ndio_hdf5_t self)
{ TRY(self);
  if(self->dataset_creation_properties<0)
    HTRY(self->dataset_creation_properties=H5Pcreate(H5P_DATASET_CREATE));
  return self->dataset_creation_properties;
Error:
  return -1;
}

static ndio_hdf5_t set_deflate(ndio_hdf5_t self)
{ hid_t out;
  HTRY(H5Pset_deflate(out=dataset_creation_properties(self),9));
  return self;
Error:
  return 0;
}

static ndio_hdf5_t set_chunk(ndio_hdf5_t self, unsigned ndim, size_t *shape)
{ hsize_t *sh;
  hid_t out;
  STACK_ALLOC(hsize_t,sh,ndim);
  reverse_hsz_sz(ndim,sh,shape);
  //sh[ndim-1]=1; // not sure what's best for chunking...this is one guess
  HTRY(H5Pset_chunk(out=dataset_creation_properties(self),ndim,sh));
  return self;
Error:
  return 0;
}

/**
 * Appends along the last dimensions.
 */
static hid_t make_dataset(ndio_hdf5_t self,nd_type_id_t typeid,unsigned ndim,size_t *shape)
{ hsize_t *sh=0;
  TRY(self->dataset==-1); // how to handle when dataset is already created?
  TRY(self->isw);
  STACK_ALLOC(hsize_t,sh,ndim);
  if(self->dataset>=0) // data set already exists...needs extending
  { HTRY(H5Sget_simple_extent_dims(space(self),sh,NULL));
    sh[ndim-1]+=shape[ndim-1]; /// FIXME! - our dimension ordering and hdf5's are opposite
  } else
  { HTRY(self->dataset=H5Dcreate(
                       self->file,name(self),
                       nd_to_hdf5_type(typeid),
                       make_space(self,ndim,shape),
                       H5P_DEFAULT,/*(rare) link creation props*/
                       dataset_creation_properties(
                          set_deflate(
                          set_chunk(self,ndim,shape))),
                       H5P_DEFAULT /*(rare) dataset access props*/
                       ));
    reverse_hsz_sz(ndim,sh,shape);
  }
  HTRY(H5Dset_extent(self->dataset,sh));
  return self->dataset;
Error:
  return -1;
}


static unsigned parse_mode_string(const char* mode,char *isr, char *isw)
{ char* c;
  char r,w,x;
  *isr=*isw=r=w=x=0;
  for(c=(char*)mode;*c;++c)
  { switch(*c)
    { case 'r': r=1; break;
      case 'w': w=1; break;
      case 'x': x=1; break;
      default:
        FAIL("Could not parse mode string.");
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
{ char isr,isw,*e,**ext;
  char **exts;
  parse_mode_string(mode,&isr,&isw);
  if(isr) 
    return H5Fis_hdf5(path);
  e=strrchr(path,'.');
  exts=(isw)?(char**)g_writeable_exts:(char**)g_readable_exts;
  for(ext=exts;*ext;++ext)
    if(strcmp(e,*ext)==0) return 1;
  return 0;
}

/**
 * Open.
 */
static void* hdf5_open(const char* path, const char* mode)
{ ndio_hdf5_t self=0;
  unsigned modeflags;
  NEW(struct _ndio_hdf5_t,self,1);
  init(self);
  modeflags=parse_mode_string(mode,&self->isr,&self->isw);
  if(!self->isr && self->isw)
    HTRY(self->file=H5Fcreate(path,        // filename
                             modeflags,    // mode flag (unsigned)
                             H5P_DEFAULT,  // creation property list
                             H5P_DEFAULT));// access property list
  else if(self->isr)
    HTRY(self->file=H5Fopen(path,modeflags,H5P_DEFAULT)); // uses file access property list
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

static nd_t hdf5_shape(ndio_t file)
{ hid_t s;
  nd_t out=0;
  unsigned ndims;
  hsize_t *sh=0;
  ndio_hdf5_t self=(ndio_hdf5_t)ndioContext(file);
  TRY(self->isr);
  TRY((s=space(self))>-1);
  TRY(out=ndinit());
  ndcast(out,hdf5_to_nd_type(dtype(self)));
  HTRY(ndims=H5Sget_simple_extent_ndims(space(self)));
  STACK_ALLOC(hsize_t,sh,ndims);
  HTRY(H5Sget_simple_extent_dims(space(self),sh,NULL));
  { unsigned i;
    for(i=0;i<ndims;++i)
      ndShapeSet(out,ndims-1-i,sh[i]);
  }
  return out;
Error:
  ndfree(out);
  return 0;
}

static unsigned hdf5_read(ndio_t file, nd_t dst)
{ ndio_hdf5_t self=(ndio_hdf5_t)ndioContext(file);
  HTRY(H5Dread(dataset(self),dtype(self),H5S_ALL,H5S_ALL,H5P_DEFAULT,nddata(dst)));
  return 1;
Error:
  return 0;
}

/**
 * Appends on last dimension.
 */
static unsigned hdf5_write(ndio_t file, nd_t src)
{ ndio_hdf5_t self=(ndio_hdf5_t)ndioContext(file);
  HTRY(H5Dwrite(make_dataset(self,ndtype(src),ndndim(src),ndshape(src)),
                nd_to_hdf5_type(ndtype(src)),
                H5S_ALL,    /*mem-space  selector*/
                H5S_ALL,    /*file-space selector*/
                H5P_DEFAULT,/*xfer props*/
                nddata(src)
                ));
  return 1;
Error:
  return 0;
}

static unsigned hdf5_subarray(ndio_t file,nd_t dst,size_t *pos,size_t *step)
{ ndio_hdf5_t self=(ndio_hdf5_t)ndioContext(file);
  hsize_t *pos_,*shape_,*step_;
  hid_t m=-1,f=-1;
  STACK_ALLOC(hsize_t,pos_,ndndim(dst));
  STACK_ALLOC(hsize_t,shape_,ndndim(dst));
  STACK_ALLOC(hsize_t,step_,ndndim(dst));
  copy_hsz_sz(ndndim(dst),shape_,ndshape(dst));
  copy_hsz_sz(ndndim(dst),pos_,pos);
  copy_hsz_sz(ndndim(dst),step_,step);

  HTRY(f=H5Dget_space(dataset(self)));
  HTRY(H5Sselect_hyperslab(f,H5S_SELECT_SET,pos_,step_,shape_,NULL/*block*/));
  HTRY(m=H5Screate_simple(ndndim(dst),shape_,NULL));
  HTRY(H5Dread(dataset(self),dtype(self),m,f,H5P_DEFAULT,nddata(dst)));
  H5Sclose(f);
  return 1;
Error:
  if(f>-1) H5Sclose(f);
  if(m>-1) H5Sclose(m);
  return 0;
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
    NULL, //canseek,
    NULL, //seek,
    hdf5_subarray,
    NULL, //add plugins
    NULL  //context
  };
  return &api;
}