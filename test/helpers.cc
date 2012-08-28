#include "helpers.h"
template<> void cast<uint8_t >(nd_t a) {ndcast(a,nd_u8 );}
template<> void cast<uint16_t>(nd_t a) {ndcast(a,nd_u16);}
template<> void cast<uint32_t>(nd_t a) {ndcast(a,nd_u32);}
template<> void cast<uint64_t>(nd_t a) {ndcast(a,nd_u64);}
template<> void cast< int8_t >(nd_t a) {ndcast(a,nd_i8 );}
template<> void cast< int16_t>(nd_t a) {ndcast(a,nd_i16);}
template<> void cast< int32_t>(nd_t a) {ndcast(a,nd_i32);}
template<> void cast< int64_t>(nd_t a) {ndcast(a,nd_i64);}
template<> void cast< float  >(nd_t a) {ndcast(a,nd_f32);}
template<> void cast< double >(nd_t a) {ndcast(a,nd_f64);}