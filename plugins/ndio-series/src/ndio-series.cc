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
   myfile.000.000.mp4
   myfile.000.001.mp4
   myfile.001.000.mp4
   myfile.001.001.mp4
   \endverbatim
 *
 * The <tt>.###.</tt> pattern represents the index on a dimension.  There are 
 * two such fields in the filenames above, and each represents a dimension.
 * Each individual file holds 3-dimensions worth of data, and two extra
 * dimensions are given by the file series.  So these files represent our 5D 
 * array.
 *
 * \author Nathan Clack
 * \date   Aug 2012
 */
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <re2/re2.h>
#include <cerrno>
#include <iostream>
#include "nd.h"

#ifdef _MSC_VER
#include "dirent.win.h"
#else
#include <dirent.h>
#endif

/// @cond DEFINES
#define PATHSEP "/"
#define countof(e) (sizeof(e)/sizeof(*e))

#define ENDL                  "\n"
#define LOG(...)              printf(__VA_ARGS__)
#define TRY(e)                do{if(!(e)) { LOG("%s(%d): %s()"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error;}} while(0)
#define TRYMSG(e,msg)         do{if(!(e)) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as false."ENDL "\t%s"ENDL "\t%sENDL",__FILE__,__LINE__,__FUNCTION__,#e,msg); goto Error; }}while(0)
#define FAIL(msg)             do{ LOG("%s(%d): %s()"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,msg); goto Error;} while(0)
#define RESIZE(type,e,nelem)  TRY((e)=(type*)realloc((e),sizeof(type)*(nelem)))
#define NEW(type,e,nelem)     TRY((e)=(type*)malloc(sizeof(type)*(nelem)))
#define SAFEFREE(e)           if(e){free(e); (e)=NULL;}
/// @endcond 

/**
 * Set read/write mode flags according to mode string.
 */
static
bool parse_mode_string(const char* mode, char *isr, char *isw)
{ const char *c=mode;
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

/** Accumulate minima in \a acc for each vector element. */
static void vmin(std::vector<unsigned> &acc, std::vector<unsigned>& pos)
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

/** Accumulate maxima in \a acc for each vector element. */
static void vmax(std::vector<unsigned> &acc, std::vector<unsigned>& pos)
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

/** Assemble full path to an ndio_t file and open it. */
static ndio_t openfile(const std::string& path, const char* fname)
{ std::string name(path);
  name.append(PATHSEP);
  name.append(fname);
  return ndioOpen(name.c_str(),NULL,"r");
}

/** Determine the shape of the array stored in the file specied by \a path
    and \a fname.
*/
static nd_t get_file_shape(const std::string& path, const char* fname)
{ nd_t shape=0; 
  ndio_t file=0;
  TRY(file=openfile(path,fname));
  shape=ndioShape(file);
  ndioClose(file);
Error:
  return shape;
}

/**
 * File context for ndio-series.
 */
struct series_t 
{
  std::string path_,     ///< the folder to search/put files
              pattern_;  ///< the filename pattern, should not include path elements
  unsigned ndim_;        ///< the number of dimensions represented in the pattern
  char     isr_,isw_;    ///< mode flags (readable, writeable)

  static RE2 ptn_field;  ///< Recognizes the "%" style filename patterns
  static RE2 eg_field;   ///< Recognizes the "*.000.000.ext" example filename patterns.

  series_t(const std::string& path, const char* mode)
  : ndim_(0)
  , isr_(0)
  , isw_(0)
  { size_t n=path.rfind(PATHSEP[0]);
    TRY(parse_mode_string(mode,&isr_,&isw_));
    { n=(n>=path.size())?0:n; // if not found set to 0
      path_=path.substr(0,n); // if PATHSEP not found will be ""
      std::string name((n==0)?path:path.substr(n+1));
      if(!gen_pattern_(name,ptn_field,"(\\\\d+)"))
        gen_pattern_(name,eg_field,".(\\\\d+)");
#if 0
      std::cout << "   PATH: "<<path_<<std::endl
                << "PATTERN: "<<pattern_<<std::endl
                << "   NDIM: "<<ndim_<<std::endl; 
#endif                
    }
  Error:
    ;
  }

  /** \returns true if series_t was opened properly, otherwise 0. */
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
   */
  bool parse(const std::string& name, std::vector<unsigned>& pos)
  { TRY(isok()); 
    { unsigned p[10];
      RE2::Arg *args[10];
      TRY(ndim_<countof(args));
      for(unsigned i=0;i<ndim_;++i)
        args[i]=new RE2::Arg(p+i);
      //^LOG("%s(%d): %s()"ENDL "%s\t%s\t%u"ENDL,__FILE__,__LINE__,__FUNCTION__,name.c_str(),pattern_.c_str(),ndim_);
      if(RE2::FullMatchN(name,pattern_,args,ndim_))
      { pos.clear();
        pos.reserve(ndim_);
        for(unsigned i=0;i<ndim_;++i)
          pos.push_back(p[i]);
        return true;
      }
      for(unsigned i=0;i<ndim_;++i)
        delete args[i];
    }
Error:
    return false;
  }

  /**
   * Generates a filename for writing corresponding to the position at \a ipos.
   * \param[out]  out   A std::string reference used for the output name.
   * \param[in]   ipos  A std::vector with the position of the filename.
   */
  bool makename(std::string& out,std::vector<size_t> &ipos)
  { char buf[128]={0};
    std::string t=pattern_;
    for (std::vector<size_t>::iterator it = ipos.begin(); it != ipos.end(); ++it)
    { snprintf(buf,countof(buf),"%zu",*it);
      TRY(RE2::Replace(&t,"\\(\\\\d\\+\\)",buf));
    }  
    out.clear();
    if(!path_.empty())
    { out+=path_;
      out+=PATHSEP;
    }
    out+=t;
    return 1;
Error:
    return 0;
  }

  /**
   * Probes the series' path for matching files and determines the maximum
   * and minimum positions indicated by the filenames.
   *
   * A minimum gets read to 0 in the corresponding dimension.
   *
   * The shape of the array along a dimension is the distance between the
   * maximum and minimum.
   *
   * \param[out] mn   A std::vector with the minima.
   * \param[out] mx   A std::vector with the maxima.
   */
  bool minmax(std::vector<unsigned>& mn, std::vector<unsigned>& mx)
  { DIR *dir=0;
    struct dirent *ent=0;
    TRYMSG(dir=opendir(path_.c_str()),strerror(errno));
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

  /** \returns the shape of the first matching file in a series as an nd_t. */
  nd_t single_file_shape()
  { DIR *dir=0;
    struct dirent *ent;
    TRYMSG(dir=opendir(path_.c_str()),strerror(errno));
    while((ent=readdir(dir))!=NULL)
    { std::vector<unsigned> pos;      
      if(parse(ent->d_name,pos))
      { nd_t shape=get_file_shape(path_,ent->d_name);
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
    { while(RE2::Replace(&name,re,repl))
        ++ndim_;
      if(ndim_) pattern_=name;
      return ndim_>0;
    }
};

/** The format name.
    Use the format name to select this format.
*/
static const char* series_fmt_name(void) { return "series"; }

/** This format is disabled for autodetection.
    \returns 0.
 */
static unsigned series_is_fmt(const char* path, const char *mode)
{
#ifdef AUTODETECT
  series_t s(path,mode);
  return s.isok();
#else
  return 0;
#endif
}

RE2 series_t::ptn_field("%+");
RE2 series_t::eg_field("\\.(\\d+)");

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
{ series_t *out=0;
  TRY(out=new series_t(path,mode));
  TRY(out->isok());
  return out;
Error:
  LOG("%s(%d): %s()"ENDL "\tCould not open"ENDL "\t\t%s"ENDL "\t\twith mode \"%s\""ENDL,
      __FILE__,__LINE__,__FUNCTION__,path?path:"(null)",mode?mode:"(null)");
  if(out) delete out;
  return 0;
}

/** Releases resources */
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
  nd_t shape=0;
  std::vector<unsigned> mn,mx;
  TRY(self->minmax(mn,mx));
  TRY(shape=self->single_file_shape());
  { size_t i,o=ndndim(shape);
    ndInsertDim(shape,o+mx.size()-1);
    for(i=0;i<mn.size();++i)
      ndShapeSet(shape,o+i,mx[i]-mn[i]+1);
  }
  return shape;
Error:
  return 0;
}

/**
 * Reads a file series into \a dst.
 */
static unsigned series_read(ndio_t file,nd_t dst)
{ series_t *self=(series_t*)ndioContext(file);
  const size_t o=ndndim(dst)-self->ndim_;
  DIR *dir;
  struct dirent *ent;
  std::vector<unsigned> mn,mx;
  TRY(self->isr_);
  TRY(self->minmax(mn,mx));
  TRYMSG(dir=opendir(self->path_.c_str()),strerror(errno));
  while((ent=readdir(dir))!=NULL)
  { std::vector<unsigned> v;
    ndio_t file=0;
    if(!self->parse(ent->d_name,v))               continue;
    if(!(file=openfile(self->path_,ent->d_name))) continue;
    for(size_t i=0;i<self->ndim_;++i) //  set the read position
      ndoffset(dst,o+i,v[i]-mn[i]);
    ndioClose(ndioRead(file,dst));
    for(size_t i=0;i<self->ndim_;++i) //reset the read position
      ndoffset(dst,o+i,-(int64_t)v[i]+(int64_t)mn[i]);
  }
  return 1;
Error:
  return 0;
}

// helpers for the write function
/// (for writing) set offset for writing a sub-array 
static void setpos(nd_t src,const size_t o,const std::vector<size_t>& ipos)
{ for(size_t i=0;i<ipos.size();++i)
    ndoffset(src,o+i,ipos[i]);
}
/// (for writing) Undo setpos() by negating the offset for writing a sub-array 
static void unsetpos(nd_t src,const size_t o,const std::vector<size_t>& ipos)
{ for(size_t i=0;i<ipos.size();++i)
    ndoffset(src,o+i,-ipos[i]);
}
/// (for writing) Maybe increment sub-array position, otherwise stop iteration.
static bool inc(nd_t src,size_t o,std::vector<size_t> &ipos)
{ int kdim=ipos.size()-1;
  while(kdim>=0 && ipos[kdim]==ndshape(src)[o+kdim]-1) // carry
    ipos[kdim--]=0;
  if(kdim<0) return 0;
  ipos[kdim]++;
#if 0
  for(size_t i=0;i<ipos.size();++i)
    printf("%5zu",ipos[i]);
  printf(ENDL);
#endif
  return 1;
}

/**
 * Write a file series.
 */
static unsigned series_write(ndio_t file, nd_t src)
{ series_t *self=(series_t*)ndioContext(file);
  size_t o;
  std::string outname;
  std::vector<size_t> ipos;
  TRY(self->isw_); // is writable?
  ipos.assign(self->ndim_,0);
  o=ndndim(src)-1;
  do
  { setpos(src,o,ipos);
    ndreshape(src,o-self->ndim_+1,ndshape(src)); // drop dimensionality
    TRY(self->makename(outname,ipos));
    ndioClose(ndioWrite(ndioOpen(outname.c_str(),NULL,"w"),src));
    ndreshape(src,o+1,ndshape(src));             // restory dimensionality
    unsetpos(src,o,ipos);
  } while (inc(src,o,ipos));
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

#include "src/io/interface.h"
/// Interface function for the ndio-series plugin.
shared const ndio_fmt_t* ndio_get_format_api(void)
{ 
  static ndio_fmt_t api=
  { series_fmt_name,
    series_is_fmt,
    series_open,
    series_close,
    series_shape,
    series_read,
    series_write,
    NULL, //set
    NULL, //get
    NULL 
  };
  // make sure init happened ok
  TRY(series_t::ptn_field.ok());
  TRY(series_t::eg_field.ok());
  return &api;
Error:
  return 0;
}