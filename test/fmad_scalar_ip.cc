/**
 * \file
 * Tests for nd_fmap_scalar_ip and ndLinearContrastAdjust_ip operations
 * \todo Do tests over different types.
 */

#include "nd.h"
#include <gtest/gtest.h>
#include "helpers.h"

struct fmad_scalar:public ::testing::Test
{ 
  
  void CPU(uint16_t init,float m,float b,uint16_t expect)
  { nd_t a;
    EXPECT_NE((void*)NULL,a=ndinit());
    EXPECT_EQ(a,ndreshapev(ndcast(a,nd_u16),3,127,63,37));
    EXPECT_EQ(a,ndref(a,malloc(ndnbytes(a)),nd_heap));
    EXPECT_EQ(a,ndfill(a,init));
    EXPECT_EQ(a,ndfmad_scalar_ip(a,m,b,0,0));
    EXPECT_EQ(-1,all<uint16_t>(ndnelem(a),(uint16_t*)nddata(a),expect));
    ndfree(a);
  }
  void GPU(uint16_t init,float mm,float bb,uint16_t expect)
  { nd_t a,b;
    EXPECT_NE((void*)NULL,a=ndinit());
    EXPECT_EQ(a,ndreshapev(ndcast(a,nd_u16),3,127,63,37));
    EXPECT_EQ(a,ndref(a,malloc(ndnbytes(a)),nd_heap));
    EXPECT_NE((void*)NULL,b=ndcuda(a,0));
    EXPECT_EQ(b,ndcopy(b,a,0,0))<<nderror(a);
    EXPECT_EQ(b,ndfill(b,init))<<nderror(b);
    EXPECT_EQ(b,ndfmad_scalar_ip(b,mm,bb,0,0))<<nderror(b);
    EXPECT_EQ(a,ndcopy(a,b,0,0))<<nderror(b);
    ndfree(b);
    EXPECT_EQ(-1,all<uint16_t>(ndnelem(a),(uint16_t*)nddata(a),expect));
    ndfree(a);
  }
};

TEST_F(fmad_scalar,CPU)               { CPU(41204, 0.5,10000.0,30602);}
TEST_F(fmad_scalar,CPU_Saturate_High) { CPU(41204, 2.0,10000.0,65535);}
TEST_F(fmad_scalar,CPU_Saturate_Low)  { CPU(41204,-2.0,10000.0,0);}
TEST_F(fmad_scalar,GPU)               { GPU(41204, 0.5,10000.0,30602);}
TEST_F(fmad_scalar,GPU_Saturate_High) { GPU(41204, 2.0,10000.0,65535);}
TEST_F(fmad_scalar,GPU_Saturate_Low)  { GPU(41204,-2.0,10000.0,0);}


TEST(LinearContrastAdjust,Test)
{ nd_t a;
  EXPECT_NE((void*)NULL,a=ndinit());
  EXPECT_EQ(a,ndreshapev(ndcast(a,nd_u16),3,127,63,37));
  EXPECT_EQ(a,ndref(a,malloc(ndnbytes(a)),nd_heap));
  EXPECT_EQ(a,ndfill(a,41204));
  EXPECT_EQ(a,ndLinearConstrastAdjust_ip(a,nd_u8,40000,50000));
  EXPECT_EQ(-1,all<uint16_t>(ndnelem(a),(uint16_t*)nddata(a),30));
  ndfree(a);
}