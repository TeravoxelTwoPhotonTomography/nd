/**
 * FFT tests.
 *
 * CUDA based implementation will fail on doubles for compute caps < 3.0
 *  - The error should be a failed plan.
 *
 * @cond TEST
 */
#include <gtest/gtest.h>
#include "helpers.h"
#include "nd.h"

#define TOL_F32 (0.5)
#define P       (12)

///// types
typedef ::testing::Types<
  float,double
  > BasicTypes;

template<class T>
struct FFTTest: public testing::Test
{
  nd_t src,dst;
  FFTTest():src(0),dst(0) {}

  virtual void SetUp()
  { ASSERT_NE((void*)NULL,src=ndinit());
    cast<T>(src);
    EXPECT_EQ(this->src,ndreshapev(this->src,2,2,1<<P ));
    EXPECT_EQ(src,ndref(src,malloc(ndnbytes(src)),nd_heap));

    { T* d=(T*)nddata(src);
      size_t n=ndnelem(src);
      for(size_t i=0;i<n;++i)
        d[i]=((unsigned)i)%127; // modulo the largest prime representable by all basic types
    }

    ASSERT_NE((void*)NULL,dst=ndheap(src));
    EXPECT_EQ(dst,ndcopy(dst,src,0,0));
  }

  void TearDown()
  {
    ndfree(src); src=0;
    ndfree(dst); dst=0;
  }
};

template<class T>
struct FFT1D: public FFTTest<T> { FFT1D() {} };

template<class T>
struct FFT2D:public FFTTest<T>
{ 
  FFT2D() {}
  void SetUp()
  { FFTTest<T>::SetUp();
    EXPECT_EQ(this->src,ndreshapev(this->src,3,2,1<<(P/2),1<<(P/2)));
    EXPECT_EQ(this->dst,ndreshapev(this->dst,3,2,1<<(P/2),1<<(P/2)));
  }
};

template<class T>
struct FFT3D:public FFTTest<T>
{ 
  FFT3D() {}
  void SetUp()
  { FFTTest<T>::SetUp();
    EXPECT_EQ(this->src,ndreshapev(this->src,4,2,1<<(P/3),1<<(P/3),1<<(P/3)));
    EXPECT_EQ(this->dst,ndreshapev(this->dst,4,2,1<<(P/3),1<<(P/3),1<<(P/3)));
  }
};

template<class T>
struct FFT4D:public FFTTest<T>
{ 
  FFT4D() {}
  void SetUp()
  { FFTTest<T>::SetUp();
    EXPECT_EQ(this->src,ndreshapev(this->src,4,2,1<<(P/4),1<<(P/4),1<<(P/4)));
    EXPECT_EQ(this->dst,ndreshapev(this->dst,4,2,1<<(P/4),1<<(P/4),1<<(P/4)));
  }
};


void dump(nd_t a, const char* name)
{ ndio_t h;
  std::cout << "DUMPIN' "<<name<<std::endl;
  if(!ndioWrite(h=ndioOpen(name,NULL,"w"),a))
    std::cout << ndioError(h) << std::endl;
  ndioClose(h);
}

template<class T> void csum(nd_t a, T& real, T& imag)
{ real=imag=0.0;
  T* data=(T*)nddata(a);
  size_t n=ndstrides(a)[ndndim(a)]/ndstrides(a)[1];
  for(size_t i=0;i<n;++i) real+=data[i*2];
  data+=1;
  for(size_t i=0;i<n;++i) imag+=data[i*2];
}

#define IDENTITY_INPLACE_C2C(type_,env_,init_) \
  TYPED_TEST(type_,Identity_InPlace_C2C_##env_) \
  { nd_fft_plan_t plan=0; \
    nd_t src=init_(this->src); \
    EXPECT_EQ(src,ndcopy(src,this->src,0,0)); \
    EXPECT_EQ(src,ndfft(src,src, 1,&plan)); \
    EXPECT_EQ(src,ndfft(src,src,-1,&plan)); \
    EXPECT_EQ(src,ndfmad_scalar_ip(src,2.0f/ndnelem(src),0.0,0,0)); \
    EXPECT_EQ(this->src,ndcopy(this->src,src,0,0)); \
    ndfftFreePlan(plan); \
    ndfree(src); \
    EXPECT_NEAR(0.0, RMSE(ndnelem(this->dst),(TypeParam*)nddata(this->dst),(TypeParam*)nddata(this->src)), TOL_F32); \
  }
#define SUM_INPLACE_C2C(type_,env_,init_) \
  TYPED_TEST(type_,Sum_InPlace_C2C_##env_) \
  { nd_t src=init_(this->src); \
    EXPECT_EQ(src,ndcopy(src,this->src,0,0)); \
    TypeParam sumr,sumi,dr,di; \
    csum(this->src,sumr,sumi); \
    EXPECT_EQ(src,ndfft(src,src, 1,NULL)); \
    EXPECT_EQ(this->src,ndcopy(this->src,src,0,0)); \
    ndfree(src); \
    dr=(sumr-((TypeParam*)nddata(this->src))[0]); \
    di=(sumi-((TypeParam*)nddata(this->src))[1]); \
    EXPECT_NEAR(0.0,sqrt(dr*dr+di*di),TOL_F32); \
  }
#define SAME_INPLACE_C2C(type_) \
  TYPED_TEST(type_,Same_InPlace_C2C) \
  { nd_t src=cuda0(this->src); \
    nd_t cpy=ndheap(this->src); \
    EXPECT_EQ(src,ndcopy(src,this->src,0,0)); \
    EXPECT_EQ(cpy,ndcopy(cpy,this->src,0,0)); \
    EXPECT_EQ(src,ndfft(src,src,1,NULL)); \
    EXPECT_EQ(cpy,ndfft(cpy,cpy,1,NULL)); \
    EXPECT_EQ(this->src,ndcopy(this->src,src,0,0)); \
    EXPECT_NEAR(0.0,RMSE(ndnelem(this->src),(TypeParam*)nddata(this->src),(TypeParam*)nddata(cpy)),TOL_F32); \
    ndfree(src); \
    ndfree(cpy); \
  }

static nd_t cuda0(nd_t a) {return ndcuda(a,0);}


TYPED_TEST_CASE(FFT1D,BasicTypes);
TYPED_TEST_CASE(FFT2D,BasicTypes);
TYPED_TEST_CASE(FFT3D,BasicTypes);
TYPED_TEST_CASE(FFT4D,BasicTypes);

IDENTITY_INPLACE_C2C(FFT1D,CPU,ndheap);
IDENTITY_INPLACE_C2C(FFT2D,CPU,ndheap);
IDENTITY_INPLACE_C2C(FFT3D,CPU,ndheap);
IDENTITY_INPLACE_C2C(FFT4D,CPU,ndheap);
IDENTITY_INPLACE_C2C(FFT2D,GPU,cuda0);
IDENTITY_INPLACE_C2C(FFT1D,GPU,cuda0);
IDENTITY_INPLACE_C2C(FFT3D,GPU,cuda0);
IDENTITY_INPLACE_C2C(FFT4D,GPU,cuda0);

SUM_INPLACE_C2C(FFT1D,CPU,ndheap);
SUM_INPLACE_C2C(FFT2D,CPU,ndheap);
SUM_INPLACE_C2C(FFT3D,CPU,ndheap);
SUM_INPLACE_C2C(FFT4D,CPU,ndheap);
SUM_INPLACE_C2C(FFT2D,GPU,cuda0);
SUM_INPLACE_C2C(FFT1D,GPU,cuda0);
SUM_INPLACE_C2C(FFT3D,GPU,cuda0);
SUM_INPLACE_C2C(FFT4D,GPU,cuda0);

SAME_INPLACE_C2C(FFT1D);
SAME_INPLACE_C2C(FFT2D);
SAME_INPLACE_C2C(FFT3D);
SAME_INPLACE_C2C(FFT4D);
/// @endcond
