/** \file
 *  Test suite for basic nd-array interface.
 *
 *  \todo Add commutativity tests for operations that should commute.
 *  \todo Add idempotence/identity tests for operations.
 *  @cond TEST
 */
#include <gtest/gtest.h>
#include "nd.h"
#include "helpers.h"

#define countof(e) (sizeof(e)/sizeof(*e))

///// types
typedef ::testing::Types<
#if 0
  uint8_t
#else
  uint8_t,uint16_t,uint32_t,uint64_t,
   int8_t, int16_t, int32_t, int64_t,
  float, double
#endif
  > BasicTypes;
  
///// Example
template<class T>
struct Example:public ::testing::Test
{
  T buf[100*100*100];
  nd_t a;

  void SetUp()
  { fill();
    a=ndinit();
    EXPECT_EQ(a,cast<T>(ndref(a,buf,nd_static))); // setting the type is only required for operations that rely on pixel arithmetic, may want to move to those tests
    EXPECT_EQ(a,ndreshapev(a,1,countof(buf)));
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

#if 0  // I removed bounds checking for reshape.
  // failed reshape
  { size_t s[] = {130,150,231};
    EXPECT_EQ((void*)NULL,ndreshape(this->a,3,s) )<<nderror(this->a);
  }
#endif
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

class InsertDim: public ::testing::Test
{ public:
  nd_t a;
  uint8_t buf[100*100*100];
  void SetUp()
  {
    size_t s[] = {20,500,100};
    ndref(a=ndinit(),buf,nd_static);
    EXPECT_NE((void*)NULL,ndreshape(a,3,s))<<nderror(a);
  }
  void TearDown()
  { ndfree(a);
  }
};

TEST_F(InsertDim,Beg)
{ size_t es[]={1,20,500,100},
         ep[]={1,1,20,10000,1000000};
  EXPECT_NE((void*)NULL,ndInsertDim(a,0));
  EXPECT_EQ(-1,firstdiff(4,es,ndshape(a)));
  EXPECT_EQ(-1,firstdiff(5,ep,ndstrides(a)));
}

TEST_F(InsertDim,End)
{ // setup shape
  size_t es[]={20,500,100,1},
         ep[]={1,20,10000,1000000,1000000};
  EXPECT_NE((void*)NULL,ndInsertDim(a,3));
  EXPECT_EQ(-1,firstdiff(4,es,ndshape(a)));
  EXPECT_EQ(-1,firstdiff(5,ep,ndstrides(a)));
}

TEST_F(InsertDim,Mid)
{ size_t es[]={20,500,1,100},
         ep[]={1,20,10000,10000,1000000};
  EXPECT_NE((void*)NULL,ndInsertDim(a,2));
  EXPECT_EQ(-1,firstdiff(4,es,ndshape(a)));
  EXPECT_EQ(-1,firstdiff(5,ep,ndstrides(a)));
}

typedef InsertDim RemoveDim;
TEST_F(RemoveDim,Beg)
{ size_t es[]={10000,100},
         ep[]={1,10000,1000000};
  EXPECT_NE((void*)NULL,ndRemoveDim(a,0));
  EXPECT_EQ(2,ndndim(a));
  EXPECT_EQ(-1,firstdiff(countof(es),es,ndshape(a)));
  EXPECT_EQ(-1,firstdiff(countof(ep),ep,ndstrides(a)));
}
TEST_F(RemoveDim,Mid)
{ size_t es[]={20,50000},
         ep[]={1,20,1000000};
  EXPECT_NE((void*)NULL,ndRemoveDim(a,1));
  EXPECT_EQ(2,ndndim(a));
  EXPECT_EQ(-1,firstdiff(countof(es),es,ndshape(a)));
  EXPECT_EQ(-1,firstdiff(countof(ep),ep,ndstrides(a)));
}
TEST_F(RemoveDim,End)
{ size_t es[]={20,500},
         ep[]={1,20,10000};
  EXPECT_NE((void*)NULL,ndRemoveDim(a,2));
  EXPECT_EQ(2,ndndim(a));
  EXPECT_EQ(-1,firstdiff(countof(es),es,ndshape(a)));
  EXPECT_EQ(-1,firstdiff(countof(ep),ep,ndstrides(a)));
}
/// @endcond
