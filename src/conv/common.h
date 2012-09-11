/**
 * \file
 * Common headers and macros for writing cuda kernels.  Do not include this file
 * in other header files.
 */
#include "../core.h"
#include "../ops.h"

#include "stdio.h"
#include <stdint.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

#include "generic/macros.h"
//define short name basic types: u8,u16,...
//  These are used for spe
TYPEDEFS;

/// @cond DEFINES

//#define DEBUG_OUTPUT
#ifdef DEBUG_OUTPUT
#define DBG(...) printf(__VA_ARGS__)
#else
#define DBG(...)
#endif

#define ENDL "\n"
#define LOG(...) ndLogError(dst_,__VA_ARGS__)
#define TRY(e)   do{if(!(e)) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e); goto Error; }}while(0)
#define CUTRY(e) do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,cudaGetErrorString(ecode)); goto Error; }}while(0)
#define FAIL     LOG("%s(%d) %s()"ENDL "\tExecution should not have reached here."ENDL,__FILE__,__LINE__,__FUNCTION__); goto Error

#ifndef restrict
#define restrict __restrict__
#endif

// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif
/// @cond DEFINES

inline __device__ float clamp(float f, float a, float b)          {return fmaxf(a, fminf(f, b));}
template<class T> inline __device__ T saturate(float f);
template<> inline __device__ uint8_t  saturate<uint8_t>(float f)  {return clamp(f,0,UCHAR_MAX);}
template<> inline __device__ uint16_t saturate<uint16_t>(float f) {return clamp(f,0,USHRT_MAX);}
template<> inline __device__ uint32_t saturate<uint32_t>(float f) {return clamp(f,0,UINT_MAX);}
template<> inline __device__ uint64_t saturate<uint64_t>(float f) {return clamp(f,0,ULLONG_MAX);} // FIXME - will overflow float type
template<> inline __device__  int8_t  saturate< int8_t> (float f) {return clamp(f,CHAR_MIN,CHAR_MAX);}
template<> inline __device__  int16_t saturate< int16_t>(float f) {return clamp(f,SHRT_MIN,SHRT_MAX);}
template<> inline __device__  int32_t saturate< int32_t>(float f) {return clamp(f,INT_MIN,INT_MAX);}
template<> inline __device__  int64_t saturate< int64_t>(float f) {return clamp(f,LLONG_MIN,LLONG_MAX);} // FIXME - will overflow float type
template<> inline __device__  float   saturate<float>(float f)    {return f;}
template<> inline __device__  double  saturate<double>(float f)   {return f;}

inline __device__ unsigned prod(dim3 a)            {return a.x*a.y*a.z;}
inline __device__ unsigned stride(uint3 a, dim3 b) {return a.x+b.x*(a.y+b.y*a.z);}
inline __device__ unsigned sum(uint3 a)            {return a.x+a.y+a.z;}