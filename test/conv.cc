/**
 * \file
 * Convolution tests.
 *
 * \todo ensure loaded test data has expected shape
 * @cond TEST
 */

#include "nd.h"
#include "config.h"
#include "helpers.h"
#include <gtest/gtest.h>

#define countof(e) (sizeof(e)/sizeof(*(e)))

//
// === SEPERABLE CONVOLUTION ===
//

#define WIDTH (256)
#define DEBUG_DUMP

static
struct _files_t
{ const char  *path;
  nd_type_id_t type;
  size_t       ndim;
  size_t       shape[3];
} file_table[] =
{
  {ND_TEST_DATA_PATH"/conv/orig.mat",nd_f32,3,{WIDTH,WIDTH,WIDTH}},
  {ND_TEST_DATA_PATH"/conv/avg0.mat",nd_f32,3,{WIDTH,WIDTH,WIDTH}},
  {ND_TEST_DATA_PATH"/conv/avg1.mat",nd_f32,3,{WIDTH,WIDTH,WIDTH}},
  {ND_TEST_DATA_PATH"/conv/avg2.mat",nd_f32,3,{WIDTH,WIDTH,WIDTH}},
  {ND_TEST_DATA_PATH"/conv/avg.mat",nd_f32,3,{WIDTH,WIDTH,WIDTH}},
  {0}
};

struct Convolve3d:public testing::Test
{ nd_t orig,avg0,avg1,avg2,avg;
  nd_t filter;
  float *data;
  static const nd_conv_params_t params;
  static const float f[];
  void SetUp(void)
  { ndioAddPluginPath(NDIO_BUILD_ROOT);
    // read in example data
    { const size_t nelem=WIDTH*WIDTH*WIDTH;
      const size_t shape[]={WIDTH,WIDTH,WIDTH};
      nd_t *as[]={&orig,&avg0,&avg1,&avg2,&avg};
      ASSERT_NE((void*)NULL,data=(float*)malloc(nelem*5*sizeof(float)));
      for(int i=0;i<countof(as);++i)
      { ndio_t file;
        ASSERT_NE((void*)NULL,*as[i]=ndinit());
        EXPECT_EQ(*as[i],ndcast(ndref(*as[i],data+nelem*i,nelem),nd_f32));
        EXPECT_EQ(*as[i],ndreshape(*as[i],3,shape));
        EXPECT_NE((void*)NULL,file=ndioOpen(file_table[i].path,NULL,"r"));
        EXPECT_EQ(file,ndioRead(file,*as[i]));
        ndioClose(file);
      }
    }
    // setup the box filter 
    { size_t shape[]={3};
      EXPECT_NE((void*)NULL,filter=ndinit());
      EXPECT_EQ(filter,ndreshape(ndcast(ndref(filter,(void*)f,3),nd_f32),1,shape));
    }
  }

  void TearDown()
  { nd_t as[]={orig,avg0,avg1,avg2,avg};
    free(data);
    for(int i=0;i<countof(as);++i)
      ndfree(as[i]);
    ndfree(filter);
  }
};
const float Convolve3d::f[]={1.0/3.0,1.0/3.0,1.0/3.0};
const nd_conv_params_t Convolve3d::params={nd_boundary_replicate};

// === CPU ===
TEST_F(Convolve3d,CPU_dim0)
{ EXPECT_EQ(orig,ndconv1_ip(orig,filter,0,&params));
#ifdef DEBUG_DUMP  
  ndioClose(ndioWrite(ndioOpen("result.tif",NULL,"w"),orig));
  ndioClose(ndioWrite(ndioOpen("expect.tif",NULL,"w"),avg0));
#endif
  EXPECT_EQ(-1,firstdiff(ndnelem(orig),(float*)nddata(orig),(float*)nddata(avg0)));
}
TEST_F(Convolve3d,CPU_dim1)
{ EXPECT_EQ(orig,ndconv1_ip(orig,filter,1,&params));
#ifdef DEBUG_DUMP
  ndioClose(ndioWrite(ndioOpen("result.tif",NULL,"w"),orig));
  ndioClose(ndioWrite(ndioOpen("expect.tif",NULL,"w"),avg1));
#endif
  EXPECT_EQ(-1,firstdiff(ndnelem(orig),(float*)nddata(orig),(float*)nddata(avg1)));
}
TEST_F(Convolve3d,CPU_dim2)
{ EXPECT_EQ(orig,ndconv1_ip(orig,filter,2,&params));
#ifdef DEBUG_DUMP
  ndioClose(ndioWrite(ndioOpen("result.tif",NULL,"w"),orig));
  ndioClose(ndioWrite(ndioOpen("expect.tif",NULL,"w"),avg2));
#endif
  EXPECT_EQ(-1,firstdiff(ndnelem(orig),(float*)nddata(orig),(float*)nddata(avg2)));
}
TEST_F(Convolve3d,CPU_alldims)
{ EXPECT_EQ(orig,ndconv1_ip(orig,filter,0,&params));
  EXPECT_EQ(orig,ndconv1_ip(orig,filter,1,&params));
  EXPECT_EQ(orig,ndconv1_ip(orig,filter,2,&params));
#ifdef DEBUG_DUMP
  ndioClose(ndioWrite(ndioOpen("result.tif",NULL,"w"),orig));
  ndioClose(ndioWrite(ndioOpen("expect.tif",NULL,"w"),avg2));
#endif
  EXPECT_EQ(-1,firstdiff(ndnelem(orig),(float*)nddata(orig),(float*)nddata(avg)));
}

// === GPU ===
TEST_F(Convolve3d,GPU_dim0)
{ nd_t dev=ndcuda(orig,0);
  EXPECT_EQ(dev,ndCudaCopy(dev,orig,0));
  EXPECT_EQ(dev,ndconv1_ip(dev,filter,0,&params));
  EXPECT_EQ(orig,ndCudaCopy(orig,dev,0));
#ifdef DEBUG_DUMP  
  ndioClose(ndioWrite(ndioOpen("result.tif",NULL,"w"),orig));
  ndioClose(ndioWrite(ndioOpen("expect.tif",NULL,"w"),avg0));
#endif
  EXPECT_EQ(-1,firstdiff(ndnelem(orig),(float*)nddata(orig),(float*)nddata(avg0))); 
}

//
// === TYPE TESTS ===
//

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

/*
  Trivial test case for doing typed-tests
 */
template<class T>
struct Convolve_1DTypeTest:public testing::Test
{   
    Convolve_1DTypeTest() {}
    void SetUp() {}
    void TearDown() {}
};

TYPED_TEST_CASE(Convolve_1DTypeTest,BasicTypes);

TYPED_TEST(Convolve_1DTypeTest,CPU)
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

TYPED_TEST(Convolve_1DTypeTest,GPU)
{ float laplace[]   ={-1/6.0,2/6.0,-1/6.0}; // sum=0, sum squares=1
  TypeParam signal[]={21,  51,   65,   84,122,100,    21, 21, //8 
                      21,  51,   65,   84,122,100,    21, 21, //16
                      21,  51,   65,   84,122,100,    21, 21, //24
                      21,  51,   65,   84,122,100,    21, 21, //32
                    };
  float     expect[]={-5,2.67,-0.83,-3.17, 10,9.5,-13.17,  0,
                      -5,2.67,-0.83,-3.17, 10,9.5,-13.17,  0,
                      -5,2.67,-0.83,-3.17, 10,9.5,-13.17,  0,
                      -5,2.67,-0.83,-3.17, 10,9.5,-13.17,  0,
                     };
  nd_t f=0,s=0;
  EXPECT_EQ(cudaSuccess,cudaSetDevice(0));
  nd_conv_params_t params={nd_boundary_replicate};
  ASSERT_NE((void*)0,f=ndinit());
  EXPECT_EQ(f,ndcast(ndref(f,laplace,countof(laplace)),nd_f32));
  EXPECT_EQ(f,ndShapeSet(f,0,countof(laplace)));
  ASSERT_NE((void*)0,s=ndinit());
  EXPECT_EQ(s,cast<TypeParam>(ndref(s,signal,countof(signal))));
  EXPECT_EQ(s,ndShapeSet(s,0,countof(signal)));

  { nd_t ff,ss;
    ASSERT_NE((void*)NULL,ss=ndcuda(s,NULL));
    EXPECT_EQ(ss,ndCudaCopy(ss,s,NULL));
    EXPECT_EQ(ss,ndconv1_ip(ss,f,0,&params))<<nderror(ss);
    EXPECT_EQ(s,ndCudaCopy(s,ss,NULL))<<nderror(s);
    ndfree(ss);
  }

  EXPECT_EQ(-1,firstdiff_clamped(countof(signal),signal,expect,0.1));

  ndfree(f);
  ndfree(s);
  EXPECT_EQ(cudaSuccess,cudaDeviceReset());
}
/// @endcond TEST