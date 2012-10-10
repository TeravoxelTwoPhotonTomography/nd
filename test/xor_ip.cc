/**
 * \file
 * Tests for fill operation
 * \todo Fill tests over different types.
 */

#include "nd.h"
#include <gtest/gtest.h>
#include "helpers.h"

TEST(XOR,CPU)
 { nd_t a;
   EXPECT_NE((void*)NULL,a=ndinit());
   EXPECT_EQ(a,ndreshapev(ndcast(a,nd_u16),3,127,63,37));
   EXPECT_EQ(a,ndref(a,malloc(ndnbytes(a)),nd_heap));
   EXPECT_EQ(a,ndfill(a,41204));
   EXPECT_EQ(a,ndxor_ip(a,38846,0,0));
   EXPECT_EQ(-1,all<uint16_t>(ndnelem(a),(uint16_t*)nddata(a),14154));
   ndfree(a);
 }

TEST(XOR,GPU)
 { nd_t a,b;
   EXPECT_NE((void*)NULL,a=ndinit());
   EXPECT_EQ(a,ndreshapev(ndcast(a,nd_u16),3,127,63,37));
   EXPECT_EQ(a,ndref(a,malloc(ndnbytes(a)),nd_heap));

   EXPECT_NE((void*)NULL,b=ndcuda(a,0));
   EXPECT_EQ(b,ndcopy(b,a,0,0))<<nderror(a);
   EXPECT_EQ(b,ndfill(b,41204))<<nderror(b);
   EXPECT_EQ(b,ndxor_ip(b,38846,0,0));
   EXPECT_EQ(a,ndcopy(a,b,0,0))<<nderror(b);
   ndfree(b);

   EXPECT_EQ(-1,all<uint16_t>(ndnelem(a),(uint16_t*)nddata(a),14154));
   ndfree(a);
 }