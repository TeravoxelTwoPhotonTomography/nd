/** \file
    Source file for instancing generic definitions.
    Beware the dark magics here.

    This is included by nd/affine/ndaffine.c.  
    This file should not be compiled on it's own.

    \author Nathan Clack
    \date   June 2012
*/
/// @cond PRIVATE

//
// TWO TYPE OPS
//
#define TSRC u8
#define TDST u8
#include "generic.affine.c"
#define TDST u16
#include "generic.affine.c"
#define TDST u32
#include "generic.affine.c"
#define TDST u64
#include "generic.affine.c"
#define TDST i8
#include "generic.affine.c"
#define TDST i16
#include "generic.affine.c"
#define TDST i32
#include "generic.affine.c"
#define TDST i64
#include "generic.affine.c"
#define TDST f32
#include "generic.affine.c"
#define TDST f64
#include "generic.affine.c"
#undef TSRC

#define TSRC u16
#define TDST u8
#include "generic.affine.c"
#define TDST u16
#include "generic.affine.c"
#define TDST u32
#include "generic.affine.c"
#define TDST u64
#include "generic.affine.c"
#define TDST i8
#include "generic.affine.c"
#define TDST i16
#include "generic.affine.c"
#define TDST i32
#include "generic.affine.c"
#define TDST i64
#include "generic.affine.c"
#define TDST f32
#include "generic.affine.c"
#define TDST f64
#include "generic.affine.c"
#undef TSRC

#define TSRC u32
#define TDST u8
#include "generic.affine.c"
#define TDST u16
#include "generic.affine.c"
#define TDST u32
#include "generic.affine.c"
#define TDST u64
#include "generic.affine.c"
#define TDST i8
#include "generic.affine.c"
#define TDST i16
#include "generic.affine.c"
#define TDST i32
#include "generic.affine.c"
#define TDST i64
#include "generic.affine.c"
#define TDST f32
#include "generic.affine.c"
#define TDST f64
#include "generic.affine.c"
#undef TSRC

#define TSRC u64
#define TDST u8
#include "generic.affine.c"
#define TDST u16
#include "generic.affine.c"
#define TDST u32
#include "generic.affine.c"
#define TDST u64
#include "generic.affine.c"
#define TDST i8
#include "generic.affine.c"
#define TDST i16
#include "generic.affine.c"
#define TDST i32
#include "generic.affine.c"
#define TDST i64
#include "generic.affine.c"
#define TDST f32
#include "generic.affine.c"
#define TDST f64
#include "generic.affine.c"
#undef TSRC

#define TSRC i8
#define TDST u8
#include "generic.affine.c"
#define TDST u16
#include "generic.affine.c"
#define TDST u32
#include "generic.affine.c"
#define TDST u64
#include "generic.affine.c"
#define TDST i8
#include "generic.affine.c"
#define TDST i16
#include "generic.affine.c"
#define TDST i32
#include "generic.affine.c"
#define TDST i64
#include "generic.affine.c"
#define TDST f32
#include "generic.affine.c"
#define TDST f64
#include "generic.affine.c"
#undef TSRC

#define TSRC i16
#define TDST u8
#include "generic.affine.c"
#define TDST u16
#include "generic.affine.c"
#define TDST u32
#include "generic.affine.c"
#define TDST u64
#include "generic.affine.c"
#define TDST i8
#include "generic.affine.c"
#define TDST i16
#include "generic.affine.c"
#define TDST i32
#include "generic.affine.c"
#define TDST i64
#include "generic.affine.c"
#define TDST f32
#include "generic.affine.c"
#define TDST f64
#include "generic.affine.c"
#undef TSRC

#define TSRC i32
#define TDST u8
#include "generic.affine.c"
#define TDST u16
#include "generic.affine.c"
#define TDST u32
#include "generic.affine.c"
#define TDST u64
#include "generic.affine.c"
#define TDST i8
#include "generic.affine.c"
#define TDST i16
#include "generic.affine.c"
#define TDST i32
#include "generic.affine.c"
#define TDST i64
#include "generic.affine.c"
#define TDST f32
#include "generic.affine.c"
#define TDST f64
#include "generic.affine.c"
#undef TSRC

#define TSRC i64
#define TDST u8
#include "generic.affine.c"
#define TDST u16
#include "generic.affine.c"
#define TDST u32
#include "generic.affine.c"
#define TDST u64
#include "generic.affine.c"
#define TDST i8
#include "generic.affine.c"
#define TDST i16
#include "generic.affine.c"
#define TDST i32
#include "generic.affine.c"
#define TDST i64
#include "generic.affine.c"
#define TDST f32
#include "generic.affine.c"
#define TDST f64
#include "generic.affine.c"
#undef TSRC

#define TSRC f32
#define TDST u8
#include "generic.affine.c"
#define TDST u16
#include "generic.affine.c"
#define TDST u32
#include "generic.affine.c"
#define TDST u64
#include "generic.affine.c"
#define TDST i8
#include "generic.affine.c"
#define TDST i16
#include "generic.affine.c"
#define TDST i32
#include "generic.affine.c"
#define TDST i64
#include "generic.affine.c"
#define TDST f32
#include "generic.affine.c"
#define TDST f64
#include "generic.affine.c"
#undef TSRC

#define TSRC f64
#define TDST u8
#include "generic.affine.c"
#define TDST u16
#include "generic.affine.c"
#define TDST u32
#include "generic.affine.c"
#define TDST u64
#include "generic.affine.c"
#define TDST i8
#include "generic.affine.c"
#define TDST i16
#include "generic.affine.c"
#define TDST i32
#include "generic.affine.c"
#define TDST i64
#include "generic.affine.c"
#define TDST f32
#include "generic.affine.c"
#define TDST f64
#include "generic.affine.c"
#undef TSRC

/// @endcode
