/** \file
 *  Tests for basic nD-array operations.
 *
 *  \todo Add xor_ip test.
 *  \todo test signed-unsigned conversion
 *  \todo Add test that makes sure inplace op's work correctly over subvolumes.
 *  \todo Add test for transpose
 *
 *  @cond TEST
 */
#include <gtest/gtest.h>
#include "nd.h"
#include "helpers.h"
#include "stdlib.h"

/// @cond DEFINES
typedef float f32;
#define TOL_F32 1e-5
#define NEW(T,e,N) EXPECT_NE((void*)NULL,(e)=(T*)malloc(sizeof(T)*N));
#define countof(e) (sizeof(e)/sizeof(*(e)))
/// @endcond

// DATA

static const f32 data01[] = {
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
}; // 5 x 3 x 4

static const f32 data02[] = {
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
}; // 5 x 3 x 4

static const f32 expt01[] = {
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,

  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,

  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
}; // 5 x 3 x 4

static const f32 expt02[] = {
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,

  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 1.0f, 0.0f, 0.0f,

  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
}; // 5 x 3 x 4

static const f32 expt03[] = {
  2.0f, 2.0f, 0.0f, 0.0f, 0.0f,
  2.0f, 2.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

  1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

  1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
  1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,

  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
}; // 5 x 3 x 4

// TESTS

class Ops3DF32:public ::testing::Test
{ public:
  size_t shape[3],strides[4];
  f32 *zeros,
      *ones,
      *e1,
      *e2,
      *e3,
      *result;
  size_t length;
  nd_t x,y,z;

  virtual void SetUp()
  { length = sizeof(data01)/sizeof(f32);
    NEW(f32,zeros,length);
    NEW(f32,ones,length);
    NEW(f32,e1,length);
    NEW(f32,e2,length);
    NEW(f32,e3,length);
    NEW(f32,result,length);
    //malloc(1024);
    memcpy(zeros,data01,sizeof(data01));
    memcpy(ones,data02,sizeof(data02));
    memcpy(e1,expt01,sizeof(expt01));
    memcpy(e2,expt02,sizeof(expt02));
    memcpy(e3,expt03,sizeof(expt03));
    ASSERT_NE((void*)NULL,x=ndinit());
    ASSERT_NE((void*)NULL,y=ndinit());
    ASSERT_NE((void*)NULL,z=ndinit());
    shape[0] = 5;
    shape[1] = 3;
    shape[2] = 4;
    size_t i;
    strides[0] = 1;
    for(i=0;i<3;++i)
      strides[i+1] = strides[i]*shape[i]; //1,5,15,60
  }

  virtual void TearDown()
  { free(zeros);
    free(ones);
    free(e1);
    free(e2);
    free(e3);
    free(result);
    ndfree(x);
    ndfree(y);
    ndfree(z);
  }
};

TEST_F(Ops3DF32,Copy)
{
  size_t sh[]  = {3,3,2};
  EXPECT_NE((void*)NULL,
    ndreshape( ndcast( ndref(x,ones,length), nd_f32),3,shape))
    <<nderror(x);
  EXPECT_NE((void*)NULL,
    ndreshape( ndcast( ndref(z,zeros,length), nd_f32),3,shape))
    <<nderror(z);
  EXPECT_NE((void*)NULL,
    ndcopy(z,x,3,sh))
    <<nderror(z);
  EXPECT_NEAR(0.0f,RMSE<f32>(strides[3],zeros,e1),TOL_F32);
}

TEST_F(Ops3DF32, Add)
{ size_t sh[] = {2,2,3};
  { nd_t   arrs[]={x,y,z};
    void  *bufs[]={e1,e2,zeros};
    for(int i=0;i<3;++i)
    EXPECT_NE((void*)NULL,
      ndreshape( ndcast( ndref(arrs[i],bufs[i],length), nd_f32),3,shape))
      <<nderror(z);
  }
  EXPECT_NE((void*)NULL,
      ndadd(z,x,y,3,sh))
      <<nderror(z);
  ASSERT_NEAR(0.0, RMSE<f32>(strides[3],zeros,e3), TOL_F32);
}

TEST_F(Ops3DF32,Cat)
{ nd_t e=0,t=0;
  // setup source array
  EXPECT_NE((void*)NULL,
    ndreshape(ndcast(ndref(e=ndinit(),(void*)expt01,length),nd_f32),3,shape))
    <<nderror(e);
  ndRemoveDim(e,1);
  // setup arg arrays
  { nd_t   arrs[]={x,y};
    size_t xsh[]={2,12},ysh[]={3,12},
           *s[]={xsh,ysh};
    for(int i=0;i<2;++i)
    { EXPECT_NE((void*)NULL,
        ndreshape(ndcast(arrs[i],nd_f32),2,s[i]))
        <<nderror(arrs[i]);
      EXPECT_NE((void*)NULL,
        ndref(arrs[i],malloc(ndnbytes(arrs[i])),ndnelem(arrs[i])))
        <<nderror(arrs[i]);
    }
    ndcopy(x,e,0,NULL);
    ndoffset(e,0,2);
    ndcopy(y,e,0,NULL);
    ndoffset(e,0,-2);
  }
  // cat and check vals
  EXPECT_NE((void*)NULL,
    t=ndcat(x,y,0))
    <<nderror(x);
  if(t)
    EXPECT_NEAR(0.0, RMSE<f32>(ndnelem(e),(f32*)nddata(e),(f32*)nddata(t)),TOL_F32);
  // cleanup
  free(nddata(x));
  free(nddata(y));
  ndfree(e);
  if(t) free(nddata(t));
  ndfree(t);
}

TEST_F(Ops3DF32,CatIP)
{ nd_t arrs[]={x,y};
  void *bufs[]={e1,e1+2*5*3};
  size_t shape[]={5,3,2};
  for(int i=0;i<2;++i)
    EXPECT_NE((void*)NULL,
      ndreshape( ndcast( ndref(arrs[i],bufs[i],length), nd_f32),3,shape))
      <<nderror(arrs[i]);
  EXPECT_NE((void*)NULL,
      ndcat_ip(x,y))
      <<nderror(x);
  /// \todo check shape and conetents
}

TEST(Ops2DU8,Transpose)
{ uint8_t buf1[]={1,2,3,4,5,6},
          buf2[]={0,0,0,0,0,0},
          expt[]={1,4,2,5,3,6};
  size_t sh1[]={3,2},sh2[]={3,2};
  nd_t dst=0,src=0;
  EXPECT_EQ(src,ndref(src=ndinit(),buf1,countof(buf1)))<<nderror(src);
  EXPECT_EQ(dst,ndref(dst=ndinit(),buf2,countof(buf2)))<<nderror(dst);
  EXPECT_EQ(src,ndreshape(src,2,sh1))<<nderror(src);
  EXPECT_EQ(dst,ndreshape(dst,2,sh2))<<nderror(dst);
  EXPECT_EQ(dst,ndtranspose(dst,src,0,1,0,NULL))<<nderror(dst);
  EXPECT_EQ(-1,firstdiff(countof(buf2),buf2,expt));
  /// \todo add checks for expected shape and stride
}

//TEST_F(Ops3DF32, Swap)
//{ size_t sh[]  = {3,3,2};
//  copy(3,sh,zeros,strides,ones,strides);
//  sh[0]=5;
//  sh[2]=1;
//  swap(3,sh,zeros+strides[2],strides,zeros+2*strides[2],strides);
//  ASSERT_NEAR(0.0f,RMSE<f32>(strides[3],zeros,e2),TOL_F32);
//}
//
//TEST_F(Ops3DF32, Gather)
//{ gather(2, 3, shape, e2, strides);
//  ASSERT_NEAR(0.0, RMSE<f32>(strides[3],e1,e2), TOL_F32);
//}
//
//TEST_F(Ops3DF32, Scatter)
//{ scatter(2, 3, shape, e1, strides);
//  ASSERT_NEAR(0.0, RMSE<f32>(strides[3],e1,e2), TOL_F32);
//}
//

/// @endcond
