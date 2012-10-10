/** \file
    Testing reading of nD volumes from file raw.

    Can't read raw files.  The format is only for writing.

    \todo APPEND test
    @cond TEST
*/

#include <gtest/gtest.h>
#include "nd.h"
#include "config.h"
    
#define countof(e) (sizeof(e)/sizeof(*e))

struct Raw: public testing::Test
{ void SetUp(void)
  { ndioAddPluginPath(NDIO_BUILD_ROOT);
  }
};


TEST_F(Raw,OpenClose)
{ // Examples that should fail to open
#if 1
  EXPECT_EQ(NULL,ndioOpen("does_not_exist.im.super.serious","raw","r"));  
  EXPECT_EQ(NULL,ndioOpen("","raw","r"));
  EXPECT_EQ(NULL,ndioOpen("","raw","w"));
  EXPECT_EQ(NULL,ndioOpen(NULL,"raw","r"));
  EXPECT_EQ(NULL,ndioOpen(NULL,"raw","w"));
#endif
  // Examples that should open
  { ndio_t file=0;
    EXPECT_NE((void*)NULL,file=ndioOpen("does_not_exist.im.super.serious","raw","w"));
    ndioClose(file);
  }
}

TEST_F(Raw,Write)
{ uint8_t data[128*128*128];
  nd_t vol=0;
  ndio_t file=0;
  EXPECT_NE((void*)NULL,vol=ndinit());
  EXPECT_EQ(vol,ndreshapev(ndcast(ndref(vol,data,nd_static),nd_u8),3,128,128,128));
  for(size_t i=0;i<countof(data);++i)
    data[i]=(uint8_t)i;
  EXPECT_NE((void*)NULL,file=ndioOpen("test.u8","raw","w"));
  EXPECT_EQ(file,ndioWrite(file,vol));
  ndioClose(file);
  ndfree(vol);
}
/// @endcond
