/**
 * Affine transform tests
 *
 * @cond TEST
 */
#include <gtest/gtest.h>
#include "helpers.h"
#include "nd.h"
#include "config.h"
#include "cuda.h"

#define TOL_F32 (1e-5)
#define NDIM    (4)
#define DIM     (64) //make big for timing, small for testing

///// types
typedef ::testing::Types<
#if 1
  uint8_t, int32_t, float
#else
  uint8_t,uint16_t,uint32_t,uint64_t,
   int8_t, int16_t, int32_t, int64_t,
  float, double
#endif
  > BasicTypes;

static void identity(float *t)
{ const int s=(NDIM+1)+1; // for addressing diagonals
  memset(t,0,sizeof(float)*(NDIM+1)*(NDIM+1));
  for(int i=0;i<(NDIM+1);++i)
    t[i*s]=1.0;
}

template<class T>
struct Affine:public testing::Test
{ nd_t src,dst;
  float *transform;
  nd_affine_params_t params;
  Affine():src(0),dst(0),transform(0) {}

  void SetUp()
  { ndioAddPluginPath(NDIO_BUILD_ROOT); // in case I want to dump any files

    ASSERT_NE((void*)NULL,transform=(float*)malloc(sizeof(float)*(NDIM+1)*(NDIM+1)));
    identity(transform);

    ASSERT_NE((void*)NULL,src=ndinit());
    cast<T>(src);
    for(int i=0;i<NDIM;++i)
      EXPECT_EQ(src,ndShapeSet(src,i,64)); // Memory: sizeof(T) * 2^(NDIM*6)
    EXPECT_EQ(src,ndref(src,malloc(ndnbytes(src)),nd_heap));

    { T* d=(T*)nddata(src);
      size_t n=ndnelem(src);
      for(size_t i=0;i<n;++i)
        d[i]=((unsigned)i)%127; // modulo the largest prime representable by all basic types
    }

    params.boundary_value=0.0;

    ASSERT_NE((void*)NULL,dst=ndinit());
    EXPECT_EQ(dst,ndreshape(ndcast(dst,\
                                   ndtype(src)),\
                            ndndim(src),\
                            ndshape(src)));
    EXPECT_EQ(dst,ndref(dst,malloc(ndnbytes(dst)),nd_heap));
    memset(nddata(dst),0,ndnbytes(dst));
  }

  void TearDown()
  {
    free(transform);
    ndfree(src);
    ndfree(dst);
  }
};

TYPED_TEST_CASE(Affine,BasicTypes);

TYPED_TEST(Affine,Identity_CPU)
{ EXPECT_EQ(this->dst,ndaffine(this->dst,this->src,this->transform,&this->params));
  EXPECT_NEAR(0.0, RMSE(ndnelem(this->dst),(TypeParam*)nddata(this->dst),(TypeParam*)nddata(this->src)), TOL_F32);
}


static void write(const char *name,nd_t a)
{ ndio_t file=ndioOpen(name,NULL,"w");
  ndioWrite(file,a);
  ndioClose(file);
}

#if 1
TYPED_TEST(Affine,Identity_GPU)
{ nd_t src_,dst_;

  { int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",device, deviceProp.major, deviceProp.minor);
    }
    cudaSetDevice(deviceCount-1); // try to avoid using the main gpu
  }

  void *xform_=0;
  { const size_t nbytes=sizeof(float)*(NDIM+1)*(NDIM+1);
    ASSERT_EQ(cudaSuccess,cudaMalloc(&xform_,nbytes));
    ASSERT_EQ(cudaSuccess,cudaMemcpy(xform_,this->transform,nbytes,cudaMemcpyHostToDevice));
  }
  //write("src-in.h5",this->src);
  //write("dst-in.h5",this->dst);
  ASSERT_NE((void*)NULL,src_=ndcuda(this->src,NULL));
  ASSERT_NE((void*)NULL,dst_=ndcuda(this->dst,NULL));
  EXPECT_EQ(dst_,ndcopy(dst_,this->dst,0,0));
  EXPECT_EQ(src_,ndcopy(src_,this->src,0,0));
  EXPECT_EQ(dst_,ndaffine(dst_,src_,(float*)xform_,&this->params))<<nderror(dst_);
  EXPECT_EQ(this->dst,ndcopy(this->dst,dst_,0,0))<<nderror(this->dst);  
  //write("src-out.h5",this->src);
  //write("dst-out.h5",this->dst);
  EXPECT_NEAR(0.0, RMSE(ndnelem(this->dst),(TypeParam*)nddata(this->dst),(TypeParam*)nddata(this->src)), TOL_F32);
  cudaFree(xform_);
  ndfree(src_);
  ndfree(dst_);
  cudaDeviceReset();
}
#endif
/// @endcond