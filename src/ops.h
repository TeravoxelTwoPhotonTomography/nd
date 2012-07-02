/** \file    
    Basic generic nd-array algorithms.

    \todo interface for kind-dependent implementations.

    \author Nathan Clack
    \date   June 2012
*/
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// Required: Include "nd.h" before this header.  
// typedef struct _nd_t* nd_t;

/** If \a ndim is 0 or \a shape is NULL, these will be
    infered from \a src and dst.

    \returns 0 on failure, otherwise \a dst
*/
nd_t ndcopy(nd_t dst, nd_t src, size_t ndim, size_t* shape);
nd_t ndadd (nd_t z, nd_t x , nd_t y, size_t ndim, size_t* shape);
nd_t ndfmad(nd_t z, float a, const nd_t x, float b, const nd_t y,size_t ndim, size_t *shape);

#ifdef __cplusplus
} //extern "C" {
#endif
