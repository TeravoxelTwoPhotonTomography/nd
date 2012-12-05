/**
 * \file
 * Tests for nd_fmap_scalar_ip and ndLinearContrastAdjust_ip operations
 * \todo Do tests over different types.
 */

#include "nd.h"
#include <gtest/gtest.h>
#include "helpers.h"

struct Saturate:public ::testing::Test
{ 
  
  void CPU(uint16_t init,uint16_t mn, uint16_t mx,uint16_t expect)
  { nd_t a;
    EXPECT_NE((void*)NULL,a=ndinit());
    EXPECT_EQ(a,ndreshapev(ndcast(a,nd_u16),3,127,63,37));
    EXPECT_EQ(a,ndref(a,malloc(ndnbytes(a)),nd_heap));
    EXPECT_EQ(a,ndfill(a,init));
    EXPECT_EQ(a,ndsaturate_ip(a,mn,mx));
    EXPECT_EQ(-1,all<uint16_t>(ndnelem(a),(uint16_t*)nddata(a),expect));
    ndfree(a);
  }
  void GPU(uint16_t init,uint16_t mn, uint16_t mx,uint16_t expect)
  { nd_t a,b;
    EXPECT_NE((void*)NULL,a=ndinit());
    EXPECT_EQ(a,ndreshapev(ndcast(a,nd_u16),3,127,63,37));
    EXPECT_EQ(a,ndref(a,malloc(ndnbytes(a)),nd_heap));
    EXPECT_NE((void*)NULL,b=ndcuda(a,0));
    EXPECT_EQ(b,ndcopy(b,a,0,0))<<nderror(a);
    EXPECT_EQ(b,ndfill(b,init))<<nderror(b);
    EXPECT_EQ(a,ndsaturate_ip(b,mn,mx))<<nderror(b);
    EXPECT_EQ(a,ndcopy(a,b,0,0))<<nderror(b);
    ndfree(b);
    EXPECT_EQ(-1,all<uint16_t>(ndnelem(a),(uint16_t*)nddata(a),expect));
    ndfree(a);
  }
};

TEST_F(Saturate,CPU)      { CPU(41204, 0     ,50000  ,41204);}
TEST_F(Saturate,CPU_High) { CPU(41204, 0     ,10000  ,10000);}
TEST_F(Saturate,CPU_Low)  { CPU(41204, 50000 ,60000  ,50000);}
TEST_F(Saturate,GPU)      { GPU(41204, 0     ,50000  ,41204);}
TEST_F(Saturate,GPU_High) { GPU(41204, 0     ,10000  ,10000);}
TEST_F(Saturate,GPU_Low)  { GPU(41204, 50000 ,60000  ,50000);}