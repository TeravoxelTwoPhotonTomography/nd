#include "config.h"
#include "nd.h"
#include "private/kind.c"

#include "stdio.h"
#include "string.h"
#include "stdlib.h"

#include "normxcorr.h"

/**
 * complex dot product
 * z=x.conj(y), where conj(y) is the complex conjugate of y
 *
 * Assume z,x,and y are all correct shape and compatibly typed.
 * Assume the first dimension in the complex one.
 */
static nd_t cdot(nd_t z, nd_t x,nd_t y)
{ //Some preconditions
  TRY(ndshape(x)[0]==2);
  TRY(ndshape(y)[0]==2);
  TRY(ndshape(z)[0]==2);
  // flip sign on y's complex values
  ndshape(ndoffset(ndPushShape(y),0,1))[0]=1;
  ndfmad_scalar_ip(y,-1,0,0,0);
  ndPopShape(y);
  ndfmad(z,x,y,0,0,0); // z=x.*y;
  // restore sign on y's complex values
  ndshape(ndoffset(ndPushShape(y),0,1))[0]=1;
  ndfmad_scalar_ip(y,-1,0,0,0);
  ndPopShape(y);
  return z;
Error:
  return 0;
}

/**
 * Expect caller to pass in real valued arrays.  Need to copy these to complex
 * valued arrays, which is the pits.  Also need the extra complex array for the
 * 3rd fft.  So memory usage for an N,M problem is 6*(M+N) intermediate, 2*(M+N)
 * for the output and M+N for the input: 9*(M+N) total.
 *
 * Could have the input off the gpu initially and direct the algorithm to
 * upload it.
 *
 * Also make sure that IFFT does the norm bc why not
 *
 * FFTs
 * 1. F1 (N)
 *    M1
 * 2. F2 (M)
 *    M2
 * 3. F(f1f1)
 *    F(f2f2)
 */