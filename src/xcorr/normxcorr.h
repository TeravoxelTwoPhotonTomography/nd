/**
 * Normalize cross correlation.
 *
 * [ ] Implement such that:
 *     If out is NULL, it will be allocated according to kind and type of the inputs.
 *     If out is NULL, the inputs are assumed to be non-padded and will be zero padded.
 * 
 */
#ifdef __cplusplus
extern "C" {
#endif

//#include "core.h"  - must be included before this file

/*
  The plan holds references to resources needed by the computation.  
  A plan can either use gpu based memory or cpu based memory (nd_heap or nd_gpu_cuda).
  The kind can be different than that used by the input/output arrays.
  The inputs ashape and bshape should have the same shape as the expected inputs
  to the "xcorr" functions before padding.

  [ ] xcorr doesn't need nearly as much memory as NCC does.
 */
typedef struct _ndxcorr_plan_t *ndxcorr_plan_t;

ndxcorr_plan_t ndxcorr_make_plan(nd_kind_t kind, nd_t ashape, nd_t bshape, uint64_t overlap_threshold);
void           ndxcorr_free_plan(ndxcorr_plan_t plan);

/*
  These assume out, a, b, maska, and maskb are the correct shapes, types and kinds.

  Use ndnormxcorr_output_shape(plan) to get the shape of the output array[1].
    
  These return out on sucess, otherwise 0.

  For the normalized cross-correlation(NCC) functions, the output has one more dimension
  than the input.  out[...,0] is the NCC,a nd out[...,1] is the number of 
  overlapping voxels contributing to the corresponding NCC value.

  [1]: The caller is responsible for ndfree'ing the returned object.
*/
nd_t ndxcorr           (nd_t out, nd_t a, nd_t b, ndxcorr_plan_t plan);
nd_t ndnormxcorr       (nd_t out, nd_t a, nd_t b, ndxcorr_plan_t plan);
nd_t ndnormxcorr_masked(nd_t out, nd_t a, nd_t maska, nd_t b, nd_t maskb, ndxcorr_plan_t plan);
nd_t ndnormxcorr_output_shape(ndxcorr_plan_t plan);

#ifdef __cplusplus
} //extern "C"
#endif
