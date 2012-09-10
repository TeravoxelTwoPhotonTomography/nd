/**
 * \file
 * Convolution tests.
 * @cond TEST
 */

#include "nd.h"
#include "helpers.h"
#include <gtest/gtest.h>

#define countof(e) (sizeof(e)/sizeof(*(e)))

///// types
typedef ::testing::Types<
#if 1
  uint8_t, int16_t, float
#else
  uint8_t,uint16_t,uint32_t,uint64_t,
   int8_t, int16_t, int32_t, int64_t,
  float, double
#endif
  > BasicTypes;

template<class T>
struct Convolve:public testing::Test
{   
    Convolve() {}
    void SetUp() {}
    void TearDown() {}
};

TYPED_TEST_CASE(Convolve,BasicTypes);

TYPED_TEST(Convolve,CPU_InPlace1D)
{ float laplace[]   ={-1/6.0,2/6.0,-1/6.0}; // sum=0, sum squares=1
  TypeParam signal[]={21,  51,   65,   84,122,100,    21, 21,   1};
  float     expect[]={-5,2.67,-0.83,-3.17, 10,9.5,-13.17,3.3,-3.3};
  nd_t f=0,s=0;
  nd_conv_params_t params={nd_boundary_replicate};
  ASSERT_NE((void*)0,f=ndinit());
  EXPECT_EQ(f,ndcast(ndref(f,laplace,countof(laplace)),nd_f32));
  EXPECT_EQ(f,ndShapeSet(f,0,countof(laplace)));
  ASSERT_NE((void*)0,s=ndinit());
  EXPECT_EQ(s,cast<TypeParam>(ndref(s,signal,countof(signal))));
  EXPECT_EQ(s,ndShapeSet(s,0,countof(signal)));

  EXPECT_EQ(s,ndconv1_ip(s,f,0,&params));
  EXPECT_EQ(-1,firstdiff_clamped(countof(signal),signal,expect,0.1));

  ndfree(f);
  ndfree(s);
}

TYPED_TEST(Convolve,GPU_InPlace1D)
{ float laplace[]   ={-1/6.0,2/6.0,-1/6.0}; // sum=0, sum squares=1
  TypeParam signal[]={21,  51,   65,   84,122,100,    21, 21,   1};
  float     expect[]={-5,2.67,-0.83,-3.17, 10,9.5,-13.17,3.3,-3.3};
  nd_t f=0,s=0;
  nd_conv_params_t params={nd_boundary_replicate};
  ASSERT_NE((void*)0,f=ndinit());
  EXPECT_EQ(f,ndcast(ndref(f,laplace,countof(laplace)),nd_f32));
  EXPECT_EQ(f,ndShapeSet(f,0,countof(laplace)));
  ASSERT_NE((void*)0,s=ndinit());
  EXPECT_EQ(s,cast<TypeParam>(ndref(s,signal,countof(signal))));
  EXPECT_EQ(s,ndShapeSet(s,0,countof(signal)));

  { nd_t ff,ss;
    ASSERT_NE((void*)NULL,ss=ndcuda(s,NULL));
    //ASSERT_NE((void*)NULL,ff=ndcuda(f,NULL));
    EXPECT_EQ(ss,ndCudaCopy(ss,s,NULL));
    //EXPECT_EQ(ff,ndCudaCopy(ff,f,NULL));
    EXPECT_EQ(ss,ndconv1_ip(ss,f,0,&params))<<nderror(ss);
    EXPECT_EQ(s,ndCudaCopy(s,ss,NULL))<<nderror(s);
    ndfree(ss);
    //ndfree(ff);
  }

  EXPECT_EQ(-1,firstdiff_clamped(countof(signal),signal,expect,0.1));

  ndfree(f);
  ndfree(s);
}
/// @endcond TEST