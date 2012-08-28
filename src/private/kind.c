/**
 * \file
 * Utilities for determining the capabilities of the different kinds of data pointers.
 *
 * Requires a macro TRYMSG to be defined.
 *
 * Defines a macro REQUIRE that checks for specified capabilites.
 *
 * This file should not be compiled.  Instead include this in implementation files that require testing
 * the capabilities of array kinds.
 *
 * The symbols defined here are really intended for internal use only.
 */

/// Kind capabilites.
#define  PTR_ARITHMETIC (1)
#define  CAN_MALLOC     (2)
#define  CAN_REALLOC    (4)
#define  CAN_FREE       (8)
#define  CAN_MEMCPY     (16)
#define  CAN_CUDA       (32)

/// Table that defines kind capabilities
static const unsigned _kind_caps[] =
{ /*nd_heap*/  PTR_ARITHMETIC | CAN_MALLOC | CAN_REALLOC | CAN_MEMCPY | CAN_FREE,
  /*nd_static*/PTR_ARITHMETIC | CAN_MEMCPY,
  /*nd_gpu*/   PTR_ARITHMETIC | CAN_CUDA,
  /*nd_file*/  0, 
};

/// Ensures the data referenced by \a a has the required capabilities.
#define REQUIRE(a,caps) TRYMSG((_kind_caps[ndkind(a)]&(caps))==(caps),#caps)