/**
 * \file
 * Helper functions for test suite.
 */
#include "helpers.h"
#include <stdint.h>
 
template<> nd_t cast<uint8_t >(nd_t a) {return ndcast(a,nd_u8 );}
template<> nd_t cast<uint16_t>(nd_t a) {return ndcast(a,nd_u16);}
template<> nd_t cast<uint32_t>(nd_t a) {return ndcast(a,nd_u32);}
template<> nd_t cast<uint64_t>(nd_t a) {return ndcast(a,nd_u64);}
template<> nd_t cast< int8_t >(nd_t a) {return ndcast(a,nd_i8 );}
template<> nd_t cast< int16_t>(nd_t a) {return ndcast(a,nd_i16);}
template<> nd_t cast< int32_t>(nd_t a) {return ndcast(a,nd_i32);}
template<> nd_t cast< int64_t>(nd_t a) {return ndcast(a,nd_i64);}
template<> nd_t cast< float  >(nd_t a) {return ndcast(a,nd_f32);}
template<> nd_t cast< double >(nd_t a) {return ndcast(a,nd_f64);}