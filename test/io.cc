#include <gtest/gtest.h>
#include "config.h"
#include "nd.h"

static 
struct _files_t
{ const char  *path;
  nd_type_id_t type;
  size_t       ndim;
  size_t       shape[5];
} 
file_table[] =
{ {ND_TEST_DATA_PATH"/vol.1ch.tif",nd_i16,3,{620,512,100,1,1}},
  {ND_TEST_DATA_PATH"/vol.rgb.tif",nd_u8 ,4,{620,512, 39,3,1}},
  {ND_TEST_DATA_PATH"/38B06.5-8.lsm",nd_u16,4,{1024,1024,248,4,1}}, // lsm's fail right now bc of the thumbnails
  {0}
};

TEST(ndio,CloseNULL) { ndioClose(NULL); }

TEST(ndio,OpenClose)
{ struct _files_t *cur;
  EXPECT_EQ(NULL,ndioOpen("does_not_exist.im.super.serious",NULL,"r"));
  for(cur=file_table;cur->path!=NULL;++cur)
  { ndio_t file=0;
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,NULL,"r"));
    ndioClose(file);
  }
}

TEST(ndio,Shape)
{ struct _files_t *cur;
  for(cur=file_table;cur->path!=NULL;++cur)
  { ndio_t file=0;
    nd_t form;
    EXPECT_NE((void*)NULL,file=ndioOpen(cur->path,NULL,"r"));
    EXPECT_NE((void*)NULL,form=ndioShape(file));
    EXPECT_EQ(cur->type,ndtype(form));
    EXPECT_EQ(cur->ndim,ndndim(form));
    for(size_t i=0;i<cur->ndim;++i)
      EXPECT_EQ(cur->shape[i],ndshape(form)[i]);
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
    EXPECT_NE((void*)NULL, vol=ndioShape(file));

    { void *data;
      EXPECT_NE((void*)NULL,data=malloc(ndnbytes(vol)));
      ndref(vol,data,ndnbytes(vol));
      EXPECT_EQ(file,ndioRead(file,vol)); // could chain ndioClose(ndioRead(ndioOpen("file.name","r"),vol));
      if(data) free(data);
    }
    ndfree(vol);
    ndioClose(file);
  }
}

TEST(ndio,WriteTiff)
{ unsigned short data[10*20*30];
  size_t shape[] = {10,20,30};
  nd_t a = ndinit();
  ndref(a,data,10*20*30);
  ndcast(a,nd_u16);
  ndreshape(a,3,shape);
  { ndio_t file=0;
    EXPECT_NE((void*)NULL,ndioWrite(ndioOpen("testout.tif",NULL,"w"),a));
    ndioClose(file);
  }
  ndfree(a);
}

TEST(ndio,MethodChainingErrors)
{ nd_t a=0;
  ndioClose(ndioRead (ndioOpen("does.not.exist",NULL,"r"),a));
  ndioClose(ndioWrite(ndioOpen("does.not.exist",NULL,"w"),a));
}