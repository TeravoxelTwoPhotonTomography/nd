#include <gtest/gtest.h>
#include "nd.h"

#define countof(e) (sizeof(e)/sizeof(*e))
///// types
typedef ::testing::Types<
#if 1
  uint8_t
#else
  uint8_t,uint16_t,uint32_t,uint64_t,
   int8_t, int16_t, int32_t, int64_t,
  float, double
#endif
  > BasicTypes;

///// helpers
template<class T> void cast(nd_t a);
template<> static void cast<uint8_t >(nd_t a) {ndcast(a,nd_u8 );}
template<> static void cast<uint16_t>(nd_t a) {ndcast(a,nd_u16);}
template<> static void cast<uint32_t>(nd_t a) {ndcast(a,nd_u32);}
template<> static void cast<uint64_t>(nd_t a) {ndcast(a,nd_u64);}
template<> static void cast< int8_t >(nd_t a) {ndcast(a,nd_i8 );}
template<> static void cast< int16_t>(nd_t a) {ndcast(a,nd_i16);}
template<> static void cast< int32_t>(nd_t a) {ndcast(a,nd_i32);}
template<> static void cast< int64_t>(nd_t a) {ndcast(a,nd_i64);}
template<> static void cast< float  >(nd_t a) {ndcast(a,nd_f32);}
template<> static void cast< double >(nd_t a) {ndcast(a,nd_f64);}

///// Example
template<class T>
struct Example:public ::testing::Test
{
  T buf[100*100*100];
  nd_t a;

  void SetUp()
  { fill();
    a=ndinit();
    ndref(a,buf,countof(buf));
    cast<T>(a); // setting the type is only required for operations that rely on pixel arithmetic, may want to move to those tests
  }
  void TearDown()
  { EXPECT_EQ(0,nderror(a));
    ndfree(a);    
  }
private:
  void fill(){for(int i=0;i<countof(buf);++i) buf[i]=i;}
};

TYPED_TEST_CASE(Example,BasicTypes);
TYPED_TEST(Example,Reshape)
{ // initial shape
  EXPECT_EQ(1,ndndim(this->a));
  EXPECT_EQ(countof(this->buf),ndshape(this->a)[0]);
  
  // failed reshape
  { size_t s[] = {13,15,231};    
    EXPECT_EQ((void*)NULL,ndreshape(this->a,3,s) )<<nderror(this->a);
  }  
  ndResetLog(this->a);
  EXPECT_EQ(1,ndndim(this->a));
  EXPECT_EQ(countof(this->buf),ndshape(this->a)[0]);

  // successful reshape
  { size_t s[] = {20,500,100};
    EXPECT_NE((void*)NULL,ndreshape(this->a,3,s) )<<nderror(this->a);
  }  
  EXPECT_EQ(3,ndndim(this->a));
  EXPECT_EQ(20,ndshape(this->a)[0]);
  EXPECT_EQ(500,ndshape(this->a)[1]);
  EXPECT_EQ(100,ndshape(this->a)[2]);
}
