/** \file
    Macros for assisting with defining and launching type dependent implementations
    from generic interfaces.

    \author Nathan Clack
    \date   June 2012
*/
#pragma once

/**
  \file
  \section MacrosSwitchCase "The switch-case pattern"

  Example:
  /code
  #define CASE2(T1,T2) dosomething_##T1##_##T2(dst,src); break
  #define CASE(T)      TYPECASE2(ndtype(dst),T); break
      TYPECASE(ndtype(src));
  #undef CASE
  #undef CASE2
  /endcode  
*/

#define TYPECASE(type_id) \
switch(type_id) \
{            \
  case nd_u8 :CASE(u8);  \
  case nd_u16:CASE(u16); \
  case nd_u32:CASE(u32); \
  case nd_u64:CASE(u64); \
  case nd_i8 :CASE(i8);  \
  case nd_i16:CASE(i16); \
  case nd_i32:CASE(i32); \
  case nd_i64:CASE(i64); \
  case nd_f32:CASE(f32); \
  case nd_f64:CASE(f64); \
  default:   \
    FAIL;    \
}

#define TYPECASE2(type_id,T) \
switch(type_id) \
{               \
  case nd_u8 :CASE2(T,u8);  \
  case nd_u16:CASE2(T,u16); \
  case nd_u32:CASE2(T,u32); \
  case nd_u64:CASE2(T,u64); \
  case nd_i8 :CASE2(T,i8);  \
  case nd_i16:CASE2(T,i16); \
  case nd_i32:CASE2(T,i32); \
  case nd_i64:CASE2(T,i64); \
  case nd_f32:CASE2(T,f32); \
  case nd_f64:CASE2(T,f64); \
  default:      \
    FAIL;       \
}
