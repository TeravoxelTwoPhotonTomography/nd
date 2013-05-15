/**
 * \file
 * Tests for FFT based cross-correlation operations.
 */

#include "nd.h"
#include <gtest/gtest.h>
#include "config.h"
#include "helpers.h"
#include <stdlib.h>
#include <string.h>

#define TOL (1e-5)

static nd_t fromfile(const char *name)
{ ndio_t f=0;
  nd_t x=ndheap_ip(ndioShape(f=ndioOpen(name,"hdf5","r")));
  ndioClose(ndioRead(f,x));
  return x;
}

// name is something like "a" or "z", buf should be big enough (ie MAX_PATH chars)
static char* filename(char *buf, size_t n, const char *name)
{ const char *dim=0,*type=0;
  memset(buf,0,n);
  strcat(buf,ND_TEST_DATA_PATH);
  strcat(buf,"/xcorr/");
  strcat(buf,name);
  strcat(buf,".mat");
  return buf;
}

class XCorrNormalizedMasked: public ::testing::Test
{ public:
  nd_t in,m1,m2,masked_ncc_overlap,masked_ncc;
  nd_t *all;  
  int  nall;
  ndxcorr_plan_t plan;

  void SetUp()
  { char buf[1024]={0};
    in=m1=m2=masked_ncc_overlap=masked_ncc=0;
    #define LOAD(v) EXPECT_NE((void*)0,(v)=fromfile(filename(buf,sizeof(buf),#v)));
    LOAD(in);
    LOAD(m1);
    LOAD(m2);
    LOAD(masked_ncc_overlap);
    LOAD(masked_ncc);
#undef LOAD
    all=&in;
    nall=5;
    plan=0;
  }
  void TearDown()
  { for(int i=0;i<nall;++i)
      ndfree(all[i]);
    ndxcorr_free_plan(plan);
  }
};
#if 0
#define DUMP(e) ndioClose(ndioWrite(ndioOpen(#e".h5",NULL,"w"),e));
#else
#define DUMP(e) 
#endif

TEST_F(XCorrNormalizedMasked,CPU)
{ EXPECT_NE((void*)0,plan=ndxcorr_make_plan(nd_heap,in,in,100));
  nd_t out=0;
  DUMP(in);
  DUMP(m1);
  DUMP(m2);
  DUMP(masked_ncc_overlap);
  DUMP(masked_ncc);
  EXPECT_NE((nd_t)0,out=ndnormxcorr_masked(0,in,m1,in,m2,plan));
  DUMP(out);
  ASSERT_EQ(out,ndPushShape(out));
  ndshape(out)[ndndim(out)-1]=1;
  EXPECT_GT(0.1,ndRMSE(masked_ncc,out));
  ndoffset(out,ndndim(out)-1,1);
  EXPECT_GT(2,ndRMSE(masked_ncc_overlap,out));
  EXPECT_EQ(out,ndPopShape(out));
  ndfree(out);
}