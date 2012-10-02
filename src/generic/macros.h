/** \file
 *  Macros for assisting with defining and launching type dependent implementations
 *  from generic interfaces.
 *
 *  Do NOT include this in another \c .h file.
 *
 *  Why use C++ templates when you could just use macros!
 *
 *  Just kidding.  Sort of.  Developing a library with an surface API that can
 *  be expressed in pure C means that any C++ templates must be hidden at some
 *  point (the advantage of a C API, as opposed to C++, is that the library can
 *  be called from another language, for example).
 *
 *  This means that one needs a mechanism for calling type specific functions
 *  from a generic interface function.  That is what these macros are intended
 *  to facilitate.
 *
 *  \section MacrosSwitchCase The switch-case pattern
 *
 *  The library supports 10 basic types.  These are described by the nd_type_id_t
 *  type.  Calling type specific implementations involves translating these id's
 *  into function calls.
 *
 *  This is accomplished using a naming convention for implementations and
 *  a macros that call these functions via switch-case statements.  Currently,
 *  functions defined up to two types are supported.
 *
 *  One type example:
 *  \code
 *  #define CASE(T)      dosomething_##T(dst,src); break
 *      TYPECASE(ndtype(src));
 *  #undef CASE
 *  \endcode
 *
 *  Two type example:
 *  \code
 *  #define CASE2(T1,T2) dosomething_##T1##_##T2(dst,src); break
 *  #define CASE(T)      TYPECASE2(ndtype(dst),T); break
 *      TYPECASE(ndtype(src));
 *  #undef CASE
 *  #undef CASE2
 *  \endcode
 *
 *  \section MacroAlternatives Other potential patterns
 *
 *  One could use function pointers to launch functions with a uniform interface.  This
 *  avoids naming conventions for the type-specific functions, but requries an interface
 *  function type to be defined for each generic operation.
 *
 *  Example:
 *  \code
 *  typedef void (*adder_impl)(nd_t a, int x); // adds x to elements of a
 *  adder_impl adders[10] = {...};             // fill with specific functions corresponding to nd_type_id's.
 *  void adder(nd_t a, int x)
 *  { if(adders[ndtype(a)])
 *      adders[ndtype(a)](a,x);
 *    else
 *      FAIL;
 *  }
 *  \endcode
 *
 *  Advantages:
 *    * a specific implementation isn't required for \a
 *      every type combination.
 *    * fits dynamically loaded implementations nicely.
 *
 *  \author Nathan Clack
 *  \date   June 2012
 */
#pragma once

/**
 * Define the short type names used for the other macros.
 */
#define TYPEDEFS \
  typedef uint8_t  u8; \
  typedef uint16_t u16;\
  typedef uint32_t u32;\
  typedef uint64_t u64;\
  typedef  int8_t  i8; \
  typedef  int16_t i16;\
  typedef  int32_t i32;\
  typedef  int64_t i64;\
  typedef  float   f32;\
  typedef  double  f64

/** Requires a macro \c CASE(T) to be defined where \c T is a type parameter.
 *  Requires a macro \c FAIL to be defined that handles when an invalid \a type_id is used.
 *  \param[in] type_id Must be a valid nd_type_id_t.
 */
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

/** Requires a macro \c CASE(T) to be defined where \c T is a type parameter.
 *  Requires a macro \c FAIL to be defined that handles when an invalid \a type_id is used.
 *  \param[in] type_id Must be a valid nd_type_id_t.
 */
#define TYPECASE_INTEGERS(type_id) \
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
  default:   \
    FAIL;    \
}

/** Requires a macro \c CASE2(T1,T2) to be defined where \c T1 and \c T2 are
 *  type parameters.
 *  Requires a macro \c FAIL to be defined that handles when an invalid \a type_id is used.
 *  \param[in] type_id Must be a valid nd_type_id_t.
 *  \param[in] T       A type name.  This should follow the u8,u16,u32,... form.  Usually
 *                     these types are defined in the implemenation function where this
 *                     macro is instanced.
 */
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
