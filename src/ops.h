/** \file
 *  Basic generic nd-array algorithms.
 *
 *  Implementations may depend on the array \a kind.
 *
 *  \todo interface for kind-dependent implementations.
 *  \todo optimize 1d core ops (sse, etc)
 *  \todo distribute over threads
 *
 *  \author Nathan Clack
 *  \date   June 2012
 *  \ingroup ndops
 */
#pragma once
#ifdef __cplusplus
extern "C" {
#endif


typedef enum boundary_condition_t_
{ nd_boundary_unknown=-1,
  nd_boundary_replicate, /// out-of-bounds values are set to the value at the nearest edge
  nd_boundary_symmetric, // [x] impl cpu    [ ] impl gpu    [ ] test
  nd_boundary_circular,  // [x] impl cpu    [ ] impl gpu    [ ] test
  nd_boundary_zero,      // [x] impl cpu    [ ] impl gpu    [ ] test
  nd_boundary_id_count
} boundary_condition_t;

typedef struct nd_affine_params_t_
{ double boundary_value;
} nd_affine_params_t;

typedef struct nd_conv_params_t_
{ boundary_condition_t boundary_condition;
} nd_conv_params_t;

// Required: Include "nd.h" before this header.  
// typedef struct _nd_t* nd_t;

nd_t ndcopy       (nd_t dst,const nd_t src,size_t ndim,size_t* shape);
nd_t ndtranspose  (nd_t dst, const nd_t src, unsigned i, unsigned j, size_t ndim, size_t *shape);
nd_t ndcat        (nd_t x,nd_t y, size_t idim);
nd_t ndcat_ip     (nd_t dst,nd_t src);
nd_t ndadd        (nd_t z,const nd_t x,const nd_t y,size_t ndim,size_t* shape);
nd_t ndfmad       (nd_t z,float a,const nd_t x,float b,const nd_t y,size_t ndim,size_t* shape);
nd_t ndxor_ip     (nd_t z,uint64_t c,size_t ndim, size_t* shape);
nd_t ndconvert_ip (nd_t z, nd_type_id_t type);

nd_t ndaffine     (nd_t dst, const nd_t src, const float *transform, const nd_affine_params_t *params);
nd_t ndconv1_ip   (nd_t dst, const nd_t filter, unsigned idim,const nd_conv_params_t* params);
#ifdef __cplusplus
} //extern "C" {
#endif
