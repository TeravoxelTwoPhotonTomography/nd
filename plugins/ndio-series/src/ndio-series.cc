/**
 * \file
 * An ndio plugin for reading file series.
 *
 * Many common image formats are only good up to two-dimensions plus a limited
 * number of colors which confounds the storage of higher dimensional data; 
 * video formats have similar problems.  However they have the advantage of 
 * being common!  It's easy to inspect those image and video files.
 *
 * This plugin helps support those formats by reading/writing a series of files
 * for the dimensions that exceed the capacity of the individual formats.  For
 * example, to a 5-dimensional array might be written to:
 *
 * \verbatim
 * myfile.000.000.mp4
 * myfile.000.001.mp4
 * myfile.001.000.mp4
 * myfile.001.001.mp4
 * \endverbatim
 *
 * The <tt>.###.</tt> pattern represents the index on a dimension.  There are 
 * two such fields in the filenames above, and each represents a dimension.
 * Each individual file holds 3-dimensions worth of data, and two extra
 * dimensions are given by the file series.  So these files represent our 5D 
 * array.
 *
 * \todo How many dimensions does a format support?
 *       Do I need to detect, or can I leave it up to the user.
 *       Should leave it up to user, write 2d or 3d tiffs?
 *
 * \todo Filename format for reading?
 *       1. Best to specify and example file: imaging picking from a gui
 *       2. Could also specify a pattern
 *          Use % sign: myfile124235%%%235%35.tif
 *          1. split string around %'s', count the number of groups
 *          2. Replace %+ patterns with (\d+)
 *          3. generate regex: myfil124235(\d+)235(\d+)35.tif
 *          4. match
 *
 * \todo Filename format for writing?
 *       1. Also specify example
 *       2. Pattern, similar to read option 2
 *       
 */
#include <string>
#include <re2/re2.h>

#define PATHSEP "/"
#define countof(e) (sizeof(e)/sizeof(*e))

namespace global
{
  RE2 ptn_field("%+"),
      eg_field("\\.(\\d+)");
}

class UnsignedArg: public RE2::Arg 
{ unsigned v_;
public:
  UnsignedArg(): RE2::Arg(&v_),v_(0) {}
  unsigned operator() const {return v_;}
};

bool parse_mode_string(const char* mode, char *isr, char *isw)
{ char *c=mode;
  *isr=*isw=0;
  do
  { switch(*c)
    { case 'r': *isr=1; break;
      case 'w': *isw=1; break;
      default:  FAIL("Invalid mode string");
    }
  } while(*++c);
  return true;
Error:
  return false;
}

static void vmin(std::vector<unsigned> &acc,const std::vector<unsigned>& pos)
{ if(acc.size()!=pos.size())
  { acc=pos;
  } else
  { std::vector<unsigned>::iterator iacc,ipos;
    for(iacc=acc.begin(),ipos=pos.begin();iacc!=acc.end();++iacc,++ipos)
    { unsigned m=*iacc,p=*ipos;
      *iacc=(m<p)?m:p;
    }
  }
}

static void vmax(std::vector<unsigned> &acc,const std::vector<unsigned>& pos)
{ if(acc.size()!=pos.size())
  { acc=pos;
  } else
  { std::vector<unsigned>::iterator iacc,ipos;
    for(iacc=acc.begin(),ipos=pos.begin();iacc!=acc.end();++iacc,++ipos)
    { unsigned m=*iacc,p=*ipos;
      *iacc=(m>p)?m:p;
    }
  }
}

static ndio_t openfile(const std::string& path, const char* fname)
{ std::string name(self->path_);
  name.append(PATHSEP);
  name.append(ent->d_name);
  return ndioOpen(name.c_str(),"r");
}

static nd_t get_file_shape(const std::string& path, const char* fname)
{ nd_t shape=0; 
  ndio_t file=0;
  TRY(file=openfile(path,fname));
  shape=ndioShape(file);
  ndioClose(file);
Error:
  return shape;
}

struct series_t 
{
  std::string path_,     ///< the folder to search/put files
              pattern_;  ///< the filename pattern, should not include path elements
  unsigned ndim_;        ///< the number of dimensions represented in the pattern
  char     isr_,isw_;    ///< mode flags (readable, writeable)

  series_t(const std::string& path, const char* mode)
  : ndim_(0)
  , isr_(0)
  , ism_(0)
  { size_t n=path.rfind(PATHSEP[0]);
    TRY(parse_mode_string(mode,&isr_,&isw_));
    n=(n==path.size())?0:n; // if not found set to 0
    path_=path.substr(0,n); // if PATHSEP not found will be ""
    std::string name((n==0)?path:path.substr(n+1));
    if(gen_pattern_(name,global::ptn_field,"(\\\\d+)")) return;
    gen_pattern_(name,global::eg_field,".(\\\\d+)");
  Error:
    ;
  }

  bool isok() { return ndim_>0; }

  /**
   * Parse \a name according to \a pattern_ to extract the position of 
   * the file according to the dimensions encoded in the filename.
   *
   * \param[in]   name  The filename to parse. Should not include the path to 
   *                    the file.
   * \param[out]  pos   The position according to the dimension fields encoded
   *                    in \a name.  Only valid if the function returns true.
   * \returns true on success, otherwise false.
   * @type {[type]}
   */
  bool parse(const std::string& name, std::vector<unsigned>& pos)
  { TRY(isok()); 
    { UnsignedArg args[10]; // I think RE2 is limited to 10 or so
      TRY(ndim_<countof(args));
      if(RE2::FullMatchN(name,pattern_,&args,ndim_))
      { pos.clear();
        pos.reserve(ndim_);
        for(unsigned i=0;i<ndim_;++i)
          pos.push_back(args[i]());
        return true;
      }
    }
Error:
    return false;
  }

  bool minmax(std::vector<unsigned>& mn, std::vector<unsigned>& mx)
  { DIR *dir=0;
    struct dirent *ent=0;
    TRYMSG(dir=opendir(path_),strerror(errno));
    while((ent=readdir(dir))!=NULL)
    { std::vector<unsigned> pos;
      if(parse(ent->d_name,pos))
      { vmin(mn,pos);
        vmax(mx,pos);
      }
    }
    closedir(dir);
    return true;
  Error:
    return false;
  }

  nd_t single_file_shape()
  { DIR *dir=0;
    struct dirent *ent;
    TRYMSG(dir=diropen(path_),strerror(errno));
    while((ent=readdir(dir))!=NULL)
    { std::vector<unsigned> pos;
      if(self->parse(ent->d_name,pos))
      { nd_t shape=get_file_shape(self->path_,ent->d_name);
        closedir(dir);
        return shape;
      }
    }
  Error:
    return 0;  
  }

  private:
    /**
     * Changes \a name if a pattern is found, but otherwise leaves it 
     * untouched.
     * \returns true if a pattern is detected, otherwise false.
     */ 
    bool gen_pattern_(std::string& name, const RE2& re, const char* repl)
    { while(RE2::Replace(&name,re,repl)
        ++ndim_;
      if(ndim_) pattern_=name;
      return ndim_>0;
    }
};

static const char* series_fmt_name(void) { return "series"; }

static unsigned series_is_fmt(const char* path, const char *mode)
{ series_t s(path,mode);
  return s.isok();
}

/**
 * Opens a file series.
 *
 * The file name has to have fields corresponding to each dimension.
 * There are two file name patterns that may be used:
 *
 * 1. An example file from the series.
 *    The filename must conform to a prescribed pattern.
 * 
 *    For example: <tt>myfile.123.45.tif</tt>.
 *    This particular file would get loaded to position <tt>(...,123,45)</tt> 
 *    (where the elipses indicate the dimensions in the tif).  The series would 
 *    look for other tif files in the same directory with the same number of 
 *    fields.
 *
 * 2. A "pattern" filename where "%" symbols are used as placeholders for the
 *    dimension fields.
 *
 *     Example: <tt>myfile.%.%.tif</tt> would find/write files like the one 
 *              in the above example.
 *
 *     Example: <tt>1231%2353%351345.mp4</tt> would find/write files like
 *              <tt>12310002353111351345.mp4</tt> with a position of
 *              <tt>(...,0,111)</tt>.
 *
 * The number of dimensions to write to a series is infered from the filename.
 * All the examples above have use two dimensions in the series.  The container
 * used for individual members of the series must be able to hold the other 
 * dimensions.  If it can't, the write will fail.
 * 
 * \param[in]   path    File name as a null terminated string.
 * \param[in]   mode    Mode string: may be "r" or "w".
 * \returns 0 on error, otherwise a file context pointer.
 */
static void* series_open(const char* path, const char* mode)
{ series_t out=new series_t(path,mode);
  TRY(out->isok());
  return &out;
Error:
  if(out) delete out;
  return 0;
}

static void series_close(ndio_t file)
{ series_t *self=(series_t*)ndioContext(file);
  delete self;
}

/**
 * Iterate over file's in the path recording min and max's for dims in names.
 * Open one to get the shape.
 */
static nd_t series_shape(ndio_t file)
{ series_t *self=(series_t*)ndioContext(file);
  std::vector<unsigned> mn,mx;
  nd_t shape=0;
  TRY(self->minmax(mn,mx));
  TRY(shape=self->get_file_shape());
  { size_t i,o=ndndim(shape);
    ndInsertDim(shape,o+mx.size()-1);
    for(i=0;i<mn.size();++i)
      ndShapeSet(o+i,mx[i]-mn[i]+1);
  }
  return shape;
Error:
  return 0;
}

static unsigned series_read(ndio_t file,nd_t dst)
{ series_t *self=(series_t*)ndioContext(file);
  const size_t o=ndndim(dst)-self->ndim_;
  DIR *dir;
  struct dirent *ent;
  TRY(self->isr_);
  TRYMSG(dir=opendir(self->path_),strerror(errno));
  while((ent=readdir(dir))!=NULL)
  { std::vector<unsigned> v;
    ndio_t file=0;
    if(!self->parse(ent->d_name,v))               continue;
    if(!(file=openfile(self->path_,ent->d_name))) continue;
    for(i=0;i<self->ndim_;++i) //  set the read position
      ndoffset(dst,o+i,v[i]);
    ndioClose(ndioRead(file,dst))
    for(i=0;i<self->ndim_;++i) //reset the read position
      ndoffset(dst,o+i,-v[i]);
  }
  return 1;
Error:
  return 0;
}

/**
 * May need to track the write cursor for appends
 * (the max on the last dimension)
 *
 * May need to track the dimensions of src to ensure
 * all pushed arrays are the same # of dimensions.
 * Also want to double check size...could delegate this to 
 * ndio.
 *
 * Need a function to generate the next filename.
 *
 * May need to iterate through subarrays of a certain dimension in src.
 *
 * Actually, a little complicated.  May need to recursively peal off outer
 * dims. 
 *
 * \todo TODO TODO TODO
 */
static unsigned series_write(ndio_t file, nd_t src)
{ series_t *self=(series_t*)ndioContext(file);
  TRY(self->isw_);
  TODO;
  return 1;
Error:
  return 0;  
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

shared const ndio_fmt_t* ndio_get_format_api(void)
{ 
  static ndio_fmt_t api=
  { series_name,
    series_is_fmt,
    series_open,
    series_close,
    series_shape,
    series_read,
    series_write,
    series_set,
    series_get,
    NULL 
  };
  // make sure init happened ok
  TRY(global::ptn_field.is_ok());
  TRY(global::eg_field.is_ok());
  return &api;
Error:
  return 0;
}