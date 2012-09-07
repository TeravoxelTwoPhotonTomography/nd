#include "helpers.h"
#include <stdint.h>
#include <float.h>

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

#define max(a,b) (((a)<(b))?(b):(a))
#define min(a,b) (((a)<(b))?(a):(b))
#define CLAMP(v,a,b) min(max(v,a),b)
template<> uint8_t  clamp(double v) { return CLAMP(v,0,CHAR_MAX);}
template<> uint16_t clamp(double v) { return CLAMP(v,0,SHRT_MAX);}
template<> uint32_t clamp(double v) { return CLAMP(v,0,LONG_MAX);}
template<> uint64_t clamp(double v) { return CLAMP(v,0,LLONG_MAX);}
template<>  int8_t  clamp(double v) { return CLAMP(v,CHAR_MIN,CHAR_MAX);}
template<>  int16_t clamp(double v) { return CLAMP(v,SHRT_MIN,CHAR_MAX);}
template<>  int32_t clamp(double v) { return CLAMP(v,LONG_MIN,CHAR_MAX);}
template<>  int64_t clamp(double v) { return CLAMP(v,LLONG_MIN,LLONG_MAX);}
template<> float    clamp(double v) { return CLAMP(v,-FLT_MAX,FLT_MAX);}
template<> double   clamp(double v) { return CLAMP(v,-DBL_MAX,DBL_MAX);}