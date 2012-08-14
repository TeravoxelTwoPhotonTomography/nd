/** \file
    Testing reading and writing of nD volumes to various file types.
    \todo append test
    \todo Write tests will fail if rgb is loaded because tiff reader loads
          colors to last dim, but ffmpeg writer assumes color is first dim.
          Need dimension annotation and transpose.
    @cond TEST
*/

#include <gtest/gtest.h>
#include "config.h"
#include "helpers.h"
#include "nd.h"

#define countof(e) (sizeof(e)/sizeof(*e))

static
struct _files_t
{ const char  *path;
  nd_type_id_t type;
  size_t       ndim;
  size_t       shape[5];
}
 
file_table[] =
{
  {ND_TEST_DATA_PATH"/vol.1ch.tif",nd_i16,3,{620,512,100,1,1}},
  {ND_TEST_DATA_PATH"/vol.rgb.tif",nd_u8 ,4,{620,512, 39,3,1}},
  {ND_TEST_DATA_PATH"/vol.rgb.mp4",nd_u8 ,4,{620,512, 39,3,1}},
  {ND_TEST_DATA_PATH"/vol.rgb.ogg",nd_u8 ,4,{620,512, 39,3,1}}, // don't know how to decode properly, strange pts's, jumps from frame 0 to frame 12
  {ND_TEST_DATA_PATH"/vol.rgb.avi",nd_u8 ,4,{620,512, 39,3,1}},
//{ND_TEST_DATA_PATH"/38B06.5-8.lsm",nd_u16,4,{1024,1024,248,4,1}}, // lsm's fail right now bc of the thumbnails
  {0}
};

TEST(ndio,CloseNULL) { ndioClose(NULL); }

TEST(ndio,OpenClose)
{ struct _files_t *cur;
  // Examples that should fail to open
  EXPECT_EQ(NULL,ndioOpen("does_not_exist.im.super.serious",NULL,"r"));
  EXPECT_EQ(NULL,ndioOpen("does_not_exist.im.super.serious",NULL,"w"));
  EXPECT_EQ(NULL,ndioOpen("",NULL,"r"));
  EXPECT_EQ(NULL,ndioOpen("",NULL,"w"));
  EXPECT_EQ(NULL,ndioOpen(NULL,NULL,"r"));
  EXPECT_EQ(NULL,ndioOpen(NULL,NULL,"w"));
  // Examples that should open
  for(cur=file_table;cur->path!=NULL;++cur)
  { ndio_t file=0;
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,NULL,"r"));
    ndioClose(file);
  }
}

TEST(ndio,Name)
{ struct _files_t *cur;
  EXPECT_EQ("(error)",ndioFormatName(NULL));
  // Examples that should open
  for(cur=file_table;cur->path!=NULL;++cur)
  { ndio_t file=0;
    const char* n;
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,NULL,"r"));
    EXPECT_NE("(error)",n=ndioFormatName(file));
    printf("%s\n",n);
    ndioClose(file);
  }
}

TEST(ndio,Get)
{ void *param;
  size_t nbytes;
  ndio_t file;
  malloc(1024);
  EXPECT_NE((void*)NULL,file=ndioOpen(file_table[0].path,"tiff/mylib","r"));
  // Get not supported for first file format
  EXPECT_EQ((void*)NULL,ndioGet(file));
  ndioClose(file);
}

TEST(ndio,Set)
{ char param[] = {1,2,3,4};
  size_t nbytes = sizeof(param);
  ndio_t file;
  EXPECT_NE((void*)NULL,file=ndioOpen(file_table[0].path,"tiff/mylib","r"));
  // Set not supported for first file format
  EXPECT_EQ((void*)NULL,ndioSet(file,(void*)param,nbytes));
  ndioClose(file);
}

TEST(ndio,Shape)
{ struct _files_t *cur;
  for(cur=file_table;cur->path!=NULL;++cur)
  { ndio_t file=0;
    nd_t form;
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,NULL,"r"))<<cur->path;
    ASSERT_NE((void*)NULL,form=ndioShape(file))<<ndioError(file)<<"\n\t"<<cur->path;
    EXPECT_EQ(cur->type,ndtype(form))<<cur->path;
    EXPECT_EQ(cur->ndim,ndndim(form))<<cur->path;
    for(size_t i=0;i<cur->ndim;++i)
      EXPECT_EQ(cur->shape[i],ndshape(form)[i])<<cur->path;
    ndfree(form);
    ndioClose(file);
  }
}

TEST(ndio,Read)
{ struct _files_t *cur;
  for(cur=file_table;cur->path!=NULL;++cur)
  { ndio_t file=0;
    nd_t vol;
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,NULL,"r"));
    ASSERT_NE((void*)NULL, vol=ndioShape(file))<<ndioError(file)<<"\n\t"<<cur->path;
    { void *data;
      EXPECT_NE((void*)NULL,data=malloc(ndnbytes(vol)));
      ndref(vol,data,ndnelem(vol));
      EXPECT_EQ(file,ndioRead(file,vol)); // could chain ndioClose(ndioRead(ndioOpen("file.name","r"),vol));
      if(data) free(data);
    }
    ndfree(vol);
    ndioClose(file);
  }
}

TEST(ndio,MethodChainingErrors)
{ nd_t a=0;
  ndioClose(ndioRead (ndioOpen("does.not.exist",NULL,"r"),a));
  ndioClose(ndioWrite(ndioOpen("does.not.exist",NULL,"w"),a));
}

class Write:public ::testing::Test
{ void *data;
public:
  nd_t a;
  ndio_t file;
  Write() :a(0),file(0),data(NULL) {}
  void SetUp()
  { ndio_t infile=0;
    ASSERT_NE((void*)NULL,infile=ndioOpen(file_table[0].path,NULL,"r"));
    ASSERT_NE((void*)NULL, a=ndioShape(infile))<<ndioError(infile)<<"\n\t"<<file_table[0].path;
    { void *data;
      ASSERT_NE((void*)NULL,data=malloc(ndnbytes(a)));
      ndref(a,data,ndnelem(a));
      ASSERT_EQ(infile,ndioRead(infile,a));
    }
    ndioClose(infile);
  }
  void TearDown()
  { ndfree(a);
    if(data) free(data);
  }
};

#define WriteTestInstance(ext) \
  TEST_F(Write,ext) \
  { nd_t vol; \
    ndio_t fin; \
    EXPECT_NE((void*)NULL,ndioWrite(file=ndioOpen("testout."#ext,NULL,"w"),a)); \
    ndioClose(file); \
    EXPECT_NE((void*)NULL,fin=ndioOpen("testout."#ext,NULL,"r")); \
    ASSERT_NE((void*)NULL, vol=ndioShape(fin))<<ndioError(fin)<<"\n\t"<<"testout."#ext; \
    ndioClose(fin); \
    { int i; \
      EXPECT_EQ(-1,i=firstdiff(ndndim(a),ndshape(a),ndshape(vol)))\
          << "\torig shape["<<i<<"]: "<< ndshape(a)[i] << "\n"  \
          << "\tread shape["<<i<<"]: "<< ndshape(vol)[i] << "\n"; \
    } \
  }
WriteTestInstance(tif);
WriteTestInstance(mp4);
WriteTestInstance(m4v);
WriteTestInstance(ogg);
WriteTestInstance(webm);
//WriteTestInstance(mov); // written file is readible, but file causes a weird crash on osx 10.8 (Aug 2012)
/// @endcond
