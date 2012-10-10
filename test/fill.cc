/**
 * \file
 * Tests for fill operation
 * \todo Fill tests over different types.
 */

#include "nd.h"
#include <gtest/gtest.h>
#include "helpers.h"

TEST(Fill,CPU)
 { nd_t a;
   EXPECT_NE((void*)NULL,a=ndinit());
   EXPECT_EQ(a,ndreshapev(ndcast(a,nd_i16),3,127,63,37));
   EXPECT_EQ(a,ndref(a,malloc(ndnbytes(a)),nd_heap));
   EXPECT_EQ(a,ndfill(a,0x8000));
   EXPECT_EQ(-1,all<int16_t>(ndnelem(a),(int16_t*)nddata(a),0x8000));
   ndfree(a);
 }

TEST(Fill,GPU)
 { nd_t a,b;
   EXPECT_NE((void*)NULL,a=ndinit());
   EXPECT_EQ(a,ndreshapev(ndcast(a,nd_i16),3,127,63,37));
   EXPECT_EQ(a,ndref(a,malloc(ndnbytes(a)),nd_heap));

   EXPECT_NE((void*)NULL,b=ndcuda(a,0));
   EXPECT_EQ(b,ndcopy(b,a,0,0))<<nderror(a);
   EXPECT_EQ(b,ndfill(b,0x8000))<<nderror(b);
   EXPECT_EQ(a,ndcopy(a,b,0,0))<<nderror(b);
   ndfree(b);

   EXPECT_EQ(-1,all<int16_t>(ndnelem(a),(int16_t*)nddata(a),0x8000));
   ndfree(a);
 }