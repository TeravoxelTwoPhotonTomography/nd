/**
 * \file
 * Tests for z=a.*x+b operation.
 *
 * [ ] fmad doesn't do saturation at the moment.
 */

#include "nd.h"
#include <gtest/gtest.h>
#include "helpers.h"
#include "config.h"
#include <stdlib.h>
#include <string.h>
#include <tr1/tuple>

#define countof(e) (sizeof(e)/sizeof(*(e)))
#define TOL (1e-5)
static nd_t fromfile(const char *name)
{ ndio_t f=0;
  nd_t x=ndheap_ip(ndioShape(f=ndioOpen(name,"hdf5","r")));
  ndioClose(ndioRead(f,x));
  return x;
}

class FMAD: public ::testing::TestWithParam< std::tr1::tuple<const char*,const char*> >
{ public:
  nd_t z,zz,a,x,b;
  nd_t all[5];

  char* filename(char *buf, size_t n, const char *name) // name is something like "a" or "z", buf should be big enough (ie MAX_PATH chars)
  { const char *dim=0,*type=0;
    memset(buf,0,n);
    strcat(buf,ND_TEST_DATA_PATH);
    strcat(buf,"/fmad/");
    strcat(buf,std::tr1::get<0>(GetParam()));
    strcat(buf,"/");
    strcat(buf,name);
    strcat(buf,"_");
    strcat(buf,std::tr1::get<1>(GetParam()));
    strcat(buf,".mat");
    return buf;
  }
  void SetUp()
  { char buf[1024]={0};
    z=a=x=b=zz=0;
    EXPECT_NE((void*)0,z=fromfile(filename(buf,sizeof(buf),"z")));
    EXPECT_NE((void*)0,a=fromfile(filename(buf,sizeof(buf),"a")));
    EXPECT_NE((void*)0,x=fromfile(filename(buf,sizeof(buf),"x")));
    EXPECT_NE((void*)0,b=fromfile(filename(buf,sizeof(buf),"b")));
    EXPECT_NE((void*)0,zz=ndheap(z));
    nd_t all_[]={z,x,a,b,zz}; // order is important
    memcpy(all,all_,sizeof(all_));
  }
  void TearDown()
  { for(int i=0;i<countof(all);++i)
      ndfree(all[i]);
  }
};

TEST_P(FMAD,CPU)
{ EXPECT_EQ(zz,ndfmad(zz,a,x,b,0,0));
  EXPECT_GT(TOL,ndRMSE(zz,z));
}
TEST_P(FMAD,GPU)
{ nd_t ts[4]={0};
  for(int i=0;i<countof(ts);++i)
  { EXPECT_NE((void*)0,ts[i]=ndcuda(all[i],0));
    ndcopy(ts[i],all[i],0,0);
  }
  EXPECT_EQ(ts[0],ndfmad(ts[0],ts[1],ts[2],ts[3],0,0));
  EXPECT_EQ(zz,ndcopy(zz,ts[0],0,0));
  EXPECT_GT(TOL,ndRMSE(zz,z));
  for(int i=0;i<countof(ts);++i)
    ndfree(ts[i]);
}
INSTANTIATE_TEST_CASE_P(Ops,FMAD,testing::Combine(
  testing::Values("1d","4d"),
  testing::Values("u16","f32")
  ));

