/*
 * Copyright (C) 2011 Florian Rathgeber, florian.rathgeber@gmail.com
 *
 * This code is licensed under the MIT License.  See the FindCUDA.cmake script
 * for the text of the license.
 *
 * Based on code by Christopher Bruns published on Stack Overflow (CC-BY):
 * http://stackoverflow.com/questions/2285185
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define ENDL "\n"
#define LOG(...) // set this to printf(__VA_ARGS__) to debug
#define CUTRY(e) do{cudaError_t ecode=(e); if(ecode!=cudaSuccess) {LOG("%s(%d): %s()"ENDL "\tExpression evaluated as failure."ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,#e,cudaGetErrorString(ecode)); goto Error; }}while(0)

int main() {
  int n,i=0;
  struct cudaDeviceProp props;
  CUTRY(cudaGetDeviceProperties(&props,0)); // only care about the first one
  printf("%d%d", (int)props.major,(int)props.minor); /* this output will be parsed by FindCUDA.cmake */
  return 0;
Error:
  return 1;
}
