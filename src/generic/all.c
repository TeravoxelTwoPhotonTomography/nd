/** \file
    Source file for instancing generic definitions.
    Beware the dark magics here.

    This is included by nd/ops.c.  This file should not be compiled on it's own.

    \author Nathan Clack
    \date   June 2012
*/
/// @cond PRIVATE

//
// ONE TYPE OPS
//

///// XOR IN-PLACE
#define TDST u8
#include "ops.xor_ip.c"
#define TDST u16
#include "ops.xor_ip.c"
#define TDST u32
#include "ops.xor_ip.c"
#define TDST u64
#include "ops.xor_ip.c"
#define TDST i8
#include "ops.xor_ip.c"
#define TDST i16
#include "ops.xor_ip.c"
#define TDST i32
#include "ops.xor_ip.c"
#define TDST i64
#include "ops.xor_ip.c"
#define TSPECIAL
#include "ops.xor_ip.c"

//
// TWO TYPE OPS
//
#define TSRC u8
#define TDST u8
#include "ops.2type.c"
#define TDST u16
#include "ops.2type.c"
#define TDST u32
#include "ops.2type.c"
#define TDST u64
#include "ops.2type.c"
#define TDST i8
#include "ops.2type.c"
#define TDST i16
#include "ops.2type.c"
#define TDST i32
#include "ops.2type.c"
#define TDST i64
#include "ops.2type.c"
#define TDST f32
#include "ops.2type.c"
#define TDST f64
#include "ops.2type.c"
#undef TSRC

#define TSRC u16
#define TDST u8
#include "ops.2type.c"
#define TDST u16
#include "ops.2type.c"
#define TDST u32
#include "ops.2type.c"
#define TDST u64
#include "ops.2type.c"
#define TDST i8
#include "ops.2type.c"
#define TDST i16
#include "ops.2type.c"
#define TDST i32
#include "ops.2type.c"
#define TDST i64
#include "ops.2type.c"
#define TDST f32
#include "ops.2type.c"
#define TDST f64
#include "ops.2type.c"
#undef TSRC

#define TSRC u32
#define TDST u8
#include "ops.2type.c"
#define TDST u16
#include "ops.2type.c"
#define TDST u32
#include "ops.2type.c"
#define TDST u64
#include "ops.2type.c"
#define TDST i8
#include "ops.2type.c"
#define TDST i16
#include "ops.2type.c"
#define TDST i32
#include "ops.2type.c"
#define TDST i64
#include "ops.2type.c"
#define TDST f32
#include "ops.2type.c"
#define TDST f64
#include "ops.2type.c"
#undef TSRC

#define TSRC u64
#define TDST u8
#include "ops.2type.c"
#define TDST u16
#include "ops.2type.c"
#define TDST u32
#include "ops.2type.c"
#define TDST u64
#include "ops.2type.c"
#define TDST i8
#include "ops.2type.c"
#define TDST i16
#include "ops.2type.c"
#define TDST i32
#include "ops.2type.c"
#define TDST i64
#include "ops.2type.c"
#define TDST f32
#include "ops.2type.c"
#define TDST f64
#include "ops.2type.c"
#undef TSRC

#define TSRC i8
#define TDST u8
#include "ops.2type.c"
#define TDST u16
#include "ops.2type.c"
#define TDST u32
#include "ops.2type.c"
#define TDST u64
#include "ops.2type.c"
#define TDST i8
#include "ops.2type.c"
#define TDST i16
#include "ops.2type.c"
#define TDST i32
#include "ops.2type.c"
#define TDST i64
#include "ops.2type.c"
#define TDST f32
#include "ops.2type.c"
#define TDST f64
#include "ops.2type.c"
#undef TSRC

#define TSRC i16
#define TDST u8
#include "ops.2type.c"
#define TDST u16
#include "ops.2type.c"
#define TDST u32
#include "ops.2type.c"
#define TDST u64
#include "ops.2type.c"
#define TDST i8
#include "ops.2type.c"
#define TDST i16
#include "ops.2type.c"
#define TDST i32
#include "ops.2type.c"
#define TDST i64
#include "ops.2type.c"
#define TDST f32
#include "ops.2type.c"
#define TDST f64
#include "ops.2type.c"
#undef TSRC

#define TSRC i32
#define TDST u8
#include "ops.2type.c"
#define TDST u16
#include "ops.2type.c"
#define TDST u32
#include "ops.2type.c"
#define TDST u64
#include "ops.2type.c"
#define TDST i8
#include "ops.2type.c"
#define TDST i16
#include "ops.2type.c"
#define TDST i32
#include "ops.2type.c"
#define TDST i64
#include "ops.2type.c"
#define TDST f32
#include "ops.2type.c"
#define TDST f64
#include "ops.2type.c"
#undef TSRC

#define TSRC i64
#define TDST u8
#include "ops.2type.c"
#define TDST u16
#include "ops.2type.c"
#define TDST u32
#include "ops.2type.c"
#define TDST u64
#include "ops.2type.c"
#define TDST i8
#include "ops.2type.c"
#define TDST i16
#include "ops.2type.c"
#define TDST i32
#include "ops.2type.c"
#define TDST i64
#include "ops.2type.c"
#define TDST f32
#include "ops.2type.c"
#define TDST f64
#include "ops.2type.c"
#undef TSRC

#define TSRC f32
#define TDST u8
#include "ops.2type.c"
#define TDST u16
#include "ops.2type.c"
#define TDST u32
#include "ops.2type.c"
#define TDST u64
#include "ops.2type.c"
#define TDST i8
#include "ops.2type.c"
#define TDST i16
#include "ops.2type.c"
#define TDST i32
#include "ops.2type.c"
#define TDST i64
#include "ops.2type.c"
#define TDST f32
#include "ops.2type.c"
#define TDST f64
#include "ops.2type.c"
#undef TSRC

#define TSRC f64
#define TDST u8
#include "ops.2type.c"
#define TDST u16
#include "ops.2type.c"
#define TDST u32
#include "ops.2type.c"
#define TDST u64
#include "ops.2type.c"
#define TDST i8
#include "ops.2type.c"
#define TDST i16
#include "ops.2type.c"
#define TDST i32
#include "ops.2type.c"
#define TDST i64
#include "ops.2type.c"
#define TDST f32
#include "ops.2type.c"
#define TDST f64
#include "ops.2type.c"
#undef TSRC

/// @endcode
