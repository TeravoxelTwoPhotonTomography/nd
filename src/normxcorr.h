/**
 * Normalize cross correlation.
 *
 * 1. inputs and outputs must all have the same kind and type
 * 2. If out is NULL, it will be allocated according to kind and type of the inputs.
 * 3. Type support is determined by the fft used.  On some devices, the gpu-based
 *    fft may be for floats only.
 * 4. The masks should be boolean.  That is all values should be zero or one.
 * 5. maska must be the same shape as a.  maskb should be the same shape as maskb.
 * 
 */
#ifdef __cplusplus
extern "C" {
#endif

nd_t ndxcorr(nd_t out, nd_t a, nd_t b);
nd_t ndnormxcorr(nd_t out, nd_t a, nd_t b);
nd_t ndnormxcorr_masked(nd_t out, nd_t a, nd_t maska, nd_t b, nd_t maskb);

#ifdef __cplusplus
} //extern "C"
#endif
