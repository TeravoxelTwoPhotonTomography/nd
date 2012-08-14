/** \file
    Testing reading of nD volumes from file series.
    @cond TEST
*/

#include <gtest/gtest.h>
#include "plugins/ndio-series/config.h"
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
{ // Set a: Should be i16, but is read by mylib as u16
  {NDIO_SERIES_TEST_DATA_PATH"/a/vol.1ch%.tif"  ,nd_u16,3,{620,512,10,1,1}},
  {NDIO_SERIES_TEST_DATA_PATH"/b/vol.0.0000.tif",nd_u8 ,4,{620,512,2,16,1}},
  {0}
};

TEST(ndioSeries,OpenClose)
{ struct _files_t *cur;
  // Examples that should fail to open
#if 1
  EXPECT_EQ(NULL,ndioOpen("does_not_exist.im.super.serious","series","r"));
  EXPECT_EQ(NULL,ndioOpen("does_not_exist.im.super.serious","series","w"));
  EXPECT_EQ(NULL,ndioOpen("","series","r"));
  EXPECT_EQ(NULL,ndioOpen("","series","w"));
  EXPECT_EQ(NULL,ndioOpen(NULL,"series","r"));
  EXPECT_EQ(NULL,ndioOpen(NULL,"series","w"));
#endif
  // Examples that should open
  for(cur=file_table;cur->path!=NULL;++cur)
  { ndio_t file=0;
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,"series","r"));
    EXPECT_STREQ("series",ndioFormatName(file));
    ndioClose(file);
  }
}

TEST(ndioSeries,Shape)
{ struct _files_t *cur;
  for(cur=file_table;cur->path!=NULL;++cur)
  { ndio_t file=0;
    nd_t form;
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,"series","r"))<<cur->path;
    ASSERT_NE((void*)NULL,form=ndioShape(file))<<ndioError(file)<<"\n\t"<<cur->path;
    EXPECT_EQ(cur->type,ndtype(form))<<cur->path;
    EXPECT_EQ(cur->ndim,ndndim(form))<<cur->path;
    for(size_t i=0;i<cur->ndim;++i)
      EXPECT_EQ(cur->shape[i],ndshape(form)[i])<<cur->path;
    ndfree(form);
    ndioClose(file);
  }
}

TEST(ndioSeries,Read)
{ struct _files_t *cur;
  for(cur=file_table;cur->path!=NULL;++cur)
  { ndio_t file=0;
    nd_t vol;
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,"series","r"));
    ASSERT_NE((void*)NULL, vol=ndioShape(file))<<ndioError(file)<<"\n\t"<<cur->path;
    { void *data;
      EXPECT_NE((void*)NULL,data=malloc(ndnbytes(vol)));
      ndref(vol,data,ndnelem(vol));
      EXPECT_EQ(file,ndioRead(file,vol));
      if(data) free(data);
    }
    ndfree(vol);
    ndioClose(file);
  }
}

TEST(ndioSeries,Write)
{ 
  nd_t vol;
  // Read data
  { ndio_t file=0;
    struct _files_t *cur=file_table+1;// Open data set B
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,"series","r"));
    ASSERT_NE((void*)NULL, vol=ndioShape(file))<<ndioError(file)<<"\n\t"<<cur->path;
    { void *data;
      EXPECT_NE((void*)NULL,data=malloc(ndnbytes(vol)));
      ndref(vol,data,ndnelem(vol));
    }
    ASSERT_EQ(file,ndioRead(file,vol));
    ndioClose(file);
  }

#if 1
  // Transpose colors to last dimension
  { nd_t dst=ndinit();
    ndref(dst,malloc(ndnbytes(vol)),ndnelem(vol));
    ndreshape(ndcast(dst,ndtype(vol)),ndndim(vol),ndshape(vol));
    EXPECT_EQ(dst,ndtranspose(dst,vol,2,3,0,NULL));
    // Cleanup vol
    free(nddata(vol));  
    ndfree(vol);
    vol=dst; // Carry the array out of scope
  }
#endif

  // Write
  { ndio_t file=0;
    EXPECT_NE((void*)NULL,file=ndioOpen("B.%.tif","series","w"));
    EXPECT_NE((void*)NULL,ndioWrite(file,vol));
    ndioClose(file);
  }

  // Cleanup
  if(vol && nddata(vol)) free(nddata(vol));
  ndfree(vol);
}
/// @endcond
