/**
 * \file
 * Convolution tests.
 *
 * \todo FIXME inplace convolution requires dynamic work size (no overlapped halos between blocks)
 *             Should do out-of-place (simplest fix).
 * \todo add unrolling for common small kernel sizes
 * \todo ensure loaded test data has expected shape
 * \todo load reference data only once.
 * @cond TEST
 */
#include "cuda_runtime_api.h"
#include "nd.h"
#include "config.h"
#include "helpers.h"
#include <gtest/gtest.h>
#include <stdio.h>

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

#define countof(e) (sizeof(e)/sizeof(*(e)))

//
// === SEPERABLE CONVOLUTION ===
//

#define WIDTH (256)
//#define DEBUG_DUMP

static
struct _files_t
{ const char  *name;
  const char  *path;
  nd_type_id_t type;
  size_t       ndim;
  size_t       shape[3];
} file_table[] =
{
  {"orig",ND_TEST_DATA_PATH"/conv/orig.mat",nd_f32,3,{WIDTH,WIDTH,WIDTH}},
  {"avg0",ND_TEST_DATA_PATH"/conv/avg0.mat",nd_f32,3,{WIDTH,WIDTH,WIDTH}},
  {"avg1",ND_TEST_DATA_PATH"/conv/avg1.mat",nd_f32,3,{WIDTH,WIDTH,WIDTH}},
  {"avg2",ND_TEST_DATA_PATH"/conv/avg2.mat",nd_f32,3,{WIDTH,WIDTH,WIDTH}},
  {"avg" ,ND_TEST_DATA_PATH"/conv/avg.mat",nd_f32,3,{WIDTH,WIDTH,WIDTH}},
  {"pdim",ND_TEST_DATA_PATH"/conv/primedim.mat",nd_f32,3,{127,127,127}},
  {0}
};

struct Convolve3d:public testing::Test
{ nd_t orig,avg0,avg1,avg2,avg,pdim;
  nd_t filter;
  float *data;
  static const nd_conv_params_t params;
  static const float f[];
  void SetUp(void)
  { ndioAddPluginPath(NDIO_BUILD_ROOT);
    // read in example data
    { nd_t *as[]={&orig,&avg0,&avg1,&avg2,&avg,&pdim};
      float *d;
      int i;
      ASSERT_NE((void*)NULL,data=(float*)malloc(WIDTH*WIDTH*WIDTH*countof(as)*sizeof(float))); // one big buffer to hold all the data
      for(i=0,d=data;i<countof(as);++i)
      { ndio_t file;
        size_t nelem;
        ASSERT_NE((void*)NULL,*as[i]=ndinit());        
        EXPECT_EQ(*as[i],ndreshape(*as[i],(unsigned)file_table[i].ndim,file_table[i].shape));
        EXPECT_EQ(*as[i],ndcast(ndref(*as[i],d,nd_static),nd_f32));        
        d+=ndnelem(*as[i]);
        EXPECT_NE((void*)NULL,file=ndioOpen(file_table[i].path,NULL,"r"));
        EXPECT_EQ(file,ndioRead(file,*as[i]));
        ndioClose(file);
      }
#if 0
      for(i=0,d=data;i<countof(as);++i)
      { char name[1024]={0};
        snprintf(name,1024,"%s.tif",file_table[i].name);
        ndioClose(ndioWrite(ndioOpen(name,0,"w"),*as[i]));
      }
#endif
    }
    // setup the box filter 
    EXPECT_NE((void*)NULL,filter=ndinit());
    EXPECT_EQ(filter,ndreshapev(ndcast(ndref(filter,(void*)f,nd_static),nd_f32),1,3));
  }

  void TearDown()
  { nd_t as[]={orig,avg0,avg1,avg2,avg};
    free(data);
    for(int i=0;i<countof(as);++i)
      ndfree(as[i]);
    ndfree(filter);
  }
};
const float Convolve3d::f[]={1.0f/3.0f,1.0f/3.0f,1.0f/3.0f};
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
{ 
  EXPECT_EQ(orig,ndconv1_ip(orig,filter,0,&params));
  EXPECT_EQ(orig,ndconv1_ip(orig,filter,1,&params));
  EXPECT_EQ(orig,ndconv1_ip(orig,filter,2,&params));
#ifdef DEBUG_DUMP
  ndioClose(ndioWrite(ndioOpen("result.tif",NULL,"w"),orig));
  ndioClose(ndioWrite(ndioOpen("expect.tif",NULL,"w"),avg));
#endif
  EXPECT_EQ(-1,firstdiff(ndnelem(orig),(float*)nddata(orig),(float*)nddata(avg)));
}
TEST_F(Convolve3d,CPU_unaligned_shape)
{ nd_t sub;
  EXPECT_NE((void*)0,sub=ndheap(pdim));
  EXPECT_EQ(sub,ndcopy(sub,orig,0,0));
  EXPECT_EQ(sub,ndconv1_ip(sub,filter,0,&params));
  EXPECT_EQ(sub,ndconv1_ip(sub,filter,1,&params));
  EXPECT_EQ(sub,ndconv1_ip(sub,filter,2,&params));
#ifdef DEBUG_DUMP
  ndioClose(ndioWrite(ndioOpen("result.tif",NULL,"w"),sub));
  ndioClose(ndioWrite(ndioOpen("expect.tif",NULL,"w"),pdim));
#endif
  EXPECT_EQ(-1,firstdiff(ndnelem(sub),(float*)nddata(sub),(float*)nddata(pdim)));
  ndfree(sub);
}

// === GPU ===
TEST_F(Convolve3d,GPU_dim0)
{ nd_t src=ndcuda(orig,0),
       dst=ndcuda(orig,0);
  EXPECT_EQ(src,ndcopy(src,orig,0,0))<<nderror(src);
  EXPECT_EQ(dst,ndconv1(dst,src,filter,0,&params));
  EXPECT_EQ(orig,ndcopy(orig,dst,0,0))<<nderror(orig);;
#ifdef DEBUG_DUMP  
  ndioClose(ndioWrite(ndioOpen("result.tif",NULL,"w"),orig));
  ndioClose(ndioWrite(ndioOpen("expect.tif",NULL,"w"),avg0));
#endif
  EXPECT_EQ(-1,firstdiff(ndnelem(orig),(float*)nddata(orig),(float*)nddata(avg0)));
  EXPECT_EQ(cudaSuccess,cudaDeviceReset());
}
TEST_F(Convolve3d,GPU_dim1)
{ nd_t dst=ndcuda(orig,0),
       src=ndcuda(orig,0);
  EXPECT_EQ(src,ndcopy(src,orig,0,0))<<nderror(dst);
  EXPECT_EQ(dst,ndconv1(dst,src,filter,1,&params));
  EXPECT_EQ(orig,ndcopy(orig,dst,0,0))<<nderror(orig);
#ifdef DEBUG_DUMP  
  ndioClose(ndioWrite(ndioOpen("result.tif",NULL,"w"),orig));
  ndioClose(ndioWrite(ndioOpen("expect.tif",NULL,"w"),avg1));
#endif
  EXPECT_EQ(-1,firstdiff(ndnelem(orig),(float*)nddata(orig),(float*)nddata(avg1)));
  EXPECT_EQ(cudaSuccess,cudaDeviceReset());
}
TEST_F(Convolve3d,GPU_dim2)
{ nd_t dst=ndcuda(orig,0),
       src=ndcuda(orig,0);
  EXPECT_EQ(src,ndcopy(src,orig,0,0))<<nderror(dst);
  EXPECT_EQ(dst,ndconv1(dst,src,filter,2,&params));
  EXPECT_EQ(orig,ndcopy(orig,dst,0,0))<<nderror(orig);
#ifdef DEBUG_DUMP  
  ndioClose(ndioWrite(ndioOpen("result.tif",NULL,"w"),orig));
  ndioClose(ndioWrite(ndioOpen("expect.tif",NULL,"w"),avg2));
#endif
  EXPECT_EQ(-1,firstdiff(ndnelem(orig),(float*)nddata(orig),(float*)nddata(avg2)));
  EXPECT_EQ(cudaSuccess,cudaDeviceReset());
}
TEST_F(Convolve3d,GPU_alldims)
{ nd_t dst=ndcuda(orig,0),
       src=ndcuda(orig,0);
  EXPECT_EQ(src,ndcopy(src,orig,0,0))<<nderror(src);
  EXPECT_EQ(dst,ndconv1(dst,src,filter,0,&params));
  EXPECT_EQ(src,ndconv1(src,dst,filter,1,&params));
  EXPECT_EQ(dst,ndconv1(dst,src,filter,2,&params));
  EXPECT_EQ(orig,ndcopy(orig,dst,0,0))<<nderror(orig);
#ifdef DEBUG_DUMP  
  ndioClose(ndioWrite(ndioOpen("result.tif",NULL,"w"),orig));
  ndioClose(ndioWrite(ndioOpen("expect.tif",NULL,"w"),avg));
#endif
  EXPECT_EQ(-1,firstdiff(ndnelem(orig),(float*)nddata(orig),(float*)nddata(avg)));
  EXPECT_EQ(cudaSuccess,cudaDeviceReset());
}
TEST_F(Convolve3d,GPU_unaligned_shape)
{ for(unsigned i=0;i<ndndim(pdim);++i)
    ndshape(orig)[i]=ndshape(pdim)[i];
  nd_t dst=ndcuda(orig,0),
       src=ndcuda(orig,0);
  EXPECT_EQ(src,ndcopy(src,orig,0,0))<<nderror(src);
  EXPECT_EQ(dst,ndconv1(dst,src,filter,0,&params));
  EXPECT_EQ(src,ndconv1(src,dst,filter,1,&params));
  EXPECT_EQ(dst,ndconv1(dst,src,filter,2,&params));
  ndreshape(orig,ndndim(dst),ndshape(dst));
  EXPECT_EQ(orig,ndcopy(orig,dst,0,0))<<nderror(orig);
#ifdef DEBUG_DUMP  
  ndioClose(ndioWrite(ndioOpen("result.tif",NULL,"w"),orig));
  ndioClose(ndioWrite(ndioOpen("expect.tif",NULL,"w"),pdim));
#endif
  EXPECT_EQ(-1,firstdiff(ndnelem(orig),(float*)nddata(orig),(float*)nddata(pdim)));
  EXPECT_EQ(cudaSuccess,cudaDeviceReset());
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
  EXPECT_EQ(f,ndcast(ndref(f,laplace,nd_static),nd_f32));
  EXPECT_EQ(f,ndShapeSet(f,0,countof(laplace)));
  ASSERT_NE((void*)0,s=ndinit());
  EXPECT_EQ(s,cast<TypeParam>(ndref(s,signal,nd_static)));
  EXPECT_EQ(s,ndShapeSet(s,0,countof(signal)));

  EXPECT_EQ(s,ndconv1_ip(s,f,0,&params));
  /*  Convolve implementations should clamp saturated values rather than
      roll over.
  */
  EXPECT_EQ(-1,firstdiff_clamped(countof(signal),signal,expect,0.1));

  ndfree(f);
  ndfree(s);
}

TYPED_TEST(Convolve_1DTypeTest,GPU)
{ float laplace[]   ={-1/6.0f,2/6.0f,-1/6.0f}; // sum=0, sum squares=1
  TypeParam signal[]={21,  51,   65,   84,122,100,    21, 21, //8 
                      21,  51,   65,   84,122,100,    21, 21, //16
                      21,  51,   65,   84,122,100,    21, 21, //24
                      21,  51,   65,   84,122,100,    21, 21, //32
                    };
  float     expect[]={-5.0f,2.67f,-0.83f,-3.17f, 10.0f,9.5f,-13.17f,  0.0f,
                      -5.0f,2.67f,-0.83f,-3.17f, 10.0f,9.5f,-13.17f,  0.0f,
                      -5.0f,2.67f,-0.83f,-3.17f, 10.0f,9.5f,-13.17f,  0.0f,
                      -5.0f,2.67f,-0.83f,-3.17f, 10.0f,9.5f,-13.17f,  0.0f,
                     };
  nd_t f=0,s=0;
  EXPECT_EQ(cudaSuccess,cudaSetDevice(0));
  nd_conv_params_t params={nd_boundary_replicate};
  ASSERT_NE((void*)0,f=ndinit());
  EXPECT_EQ(f,ndcast(ndref(f,laplace,nd_static),nd_f32));
  EXPECT_EQ(f,ndShapeSet(f,0,countof(laplace)));
  ASSERT_NE((void*)0,s=ndinit());
  EXPECT_EQ(s,cast<TypeParam>(ndref(s,signal,nd_static)));
  EXPECT_EQ(s,ndShapeSet(s,0,countof(signal)));

  { nd_t ff,ss1,ss;
    ASSERT_NE((void*)NULL,ss =ndcuda(s,NULL));
    ASSERT_NE((void*)NULL,ss1=ndcuda(s,NULL));
    EXPECT_EQ(ss1,ndcopy(ss1,s,0,0));
    EXPECT_EQ(ss,ndconv1(ss,ss1,f,0,&params))<<nderror(ss1);
    EXPECT_EQ(s,ndcopy(s,ss,0,0))<<nderror(s);
    ndfree(ss);
    ndfree(ss1);
  }

  /*  Convolve implementations should clamp saturated values rather than
      roll over.
  */
  EXPECT_EQ(-1,firstdiff_clamped(countof(signal),signal,expect,0.1));

  ndfree(f);
  ndfree(s);
  EXPECT_EQ(cudaSuccess,cudaDeviceReset());
}
/// @endcond TEST