#define __STDC_LIMIT_MACROS
#include "helpers.h"
#include <stdint.h>
#include <limits.h>
#include <float.h>

  typedef uint8_t  u8; 
  typedef uint16_t u16;
  typedef uint32_t u32;
  typedef uint64_t u64;
  typedef  int8_t  i8; 
  typedef  int16_t i16;
  typedef  int32_t i32;
  typedef  int64_t i64;
  typedef  float   f32;
  typedef  double  f64;

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
template<> uint32_t clamp(double v) { return CLAMP(v,0,UINT32_MAX);}
template<> uint64_t clamp(double v) { return CLAMP(v,0,UINT64_MAX);}
template<>  int8_t  clamp(double v) { return CLAMP(v,CHAR_MIN,CHAR_MAX);}
template<>  int16_t clamp(double v) { return CLAMP(v,SHRT_MIN,CHAR_MAX);}
template<>  int32_t clamp(double v) { return CLAMP(v,INT32_MIN,INT32_MAX);}
template<>  int64_t clamp(double v) { return CLAMP(v,INT64_MIN,INT64_MAX);}
template<> float    clamp(double v) { return CLAMP(v,-FLT_MAX,FLT_MAX);}
template<> double   clamp(double v) { return CLAMP(v,-DBL_MAX,DBL_MAX);}


#define TYPECASE(type_id) \
switch(type_id) \
{            \
  case nd_u8 :CASE(u8 ); \
  case nd_u16:CASE(u16); \
  case nd_u32:CASE(u32); \
  case nd_u64:CASE(u64); \
  case nd_i8 :CASE(i8 ); \
  case nd_i16:CASE(i16); \
  case nd_i32:CASE(i32); \
  case nd_i64:CASE(i64); \
  case nd_f32:CASE(f32); \
  case nd_f64:CASE(f64); \
  default:   \
    FAIL;    \
}
#define FAIL
double ndRMSE(nd_t a, nd_t b)
{ 
#define CASE(T) return RMSE<T>(ndnelem(a),(T*)nddata(a),(T*)nddata(b));
  TYPECASE(ndtype(a));
#undef CASE
  return 999999999.0;
}