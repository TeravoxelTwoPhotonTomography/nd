/**
 * \file
 * nd convolution transform on the GPU with CUDA.
 */
#include "common.h"

#define MAX_FILTER_WIDTH     32  // max size of kernel (allocated in constant memory)

#define COLUMNS_BLOCKDIM_X   32
#define COLUMNS_BLOCKDIM_Y   8
#define COLUMNS_RESULT_STEPS 8
#define COLUMNS_HALO_STEPS   1

  /**
   * Treat our nd data as 2d around idim:
   * idim is the dimension to be convolved.
   */
struct arg_t
{ u8   *restrict data;
  unsigned       nrows,
                 ncols,
                 rstride, // strides are in elements (not bytes)
                 cstride;
  arg_t(nd_t a,unsigned idim):
    data(0),nrows(0),ncols(0),rstride(0),cstride(0)
  { 
    TRY(ndstrides(a)[0]==ndbpp(a)); // only support unit element strides
    TRY(data=nddata(a));
    cstride=1;
    if(idim==0)
    { rstride=ndstrides(a)[1]/ndstrides(a)[0];
      ncols=ndshape(a)[0];
      nrows=ndshape(a)[1];
    }
    else
    { rstride=ndstrides(a)[idim]/ndstrides(a)[0];
      ncols=ndshape(a)[0];
      nrows=ndshape(a)[idim];
    }
Error:
    return; // data will be 0 if there was an error
  }
  bool isok() { return data!=0;}
};

__constant__ float c_kernel[MAX_FILTER_WIDTH];

static unsigned upload_kernel(float *kernel,unsigned n)
{ TRY(n<MAX_FILTER_WIDTH);
  CUTRY(cudaMemcpyToSymbol(c_kernel,kernel,n*sizeof(*kernel)));
  return 1;
Error:
  return 0;
}


template<
  typename Tdst,
  unsigned BX  ,//=32,       // should be 1 warp's worth
  unsigned BY  ,//=4,        // the number of warps to schedule for a single block
  unsigned HALO,//=1,        // the number of halo elements to load per thread per side
  unsigned WORK //=8,        // each thread processes this many elements in a row
>
__global__ void 
__launch_bounds__(BX*BY,1) /*max threads,min blocks*/
  conv1_ip_rows(arg_t dst_, int radius, const nd_conv_params_t param)
{  __shared__ Tdst buf[BY][(WORK+2*HALO)*BX];
  Tdst* dst=(Tdst*)dst_.data;  
  const int ox=threadIdx.x+(blockIdx.x*WORK-HALO)*BX,
            oy=threadIdx.y+ blockIdx.y           *BY;
  dst+=ox+oy*dst_.rstride;  
#pragma unroll
  for(int i=HALO     ;i<HALO+WORK  ;++i) buf[threadIdx.y][threadIdx.x+i*BX]=dst[i*BX];
#pragma unroll
  for(int i=0        ;i<HALO       ;++i) buf[threadIdx.y][threadIdx.x+i*BX]=(ox>=-i*BX)          ?dst[i*BX]:dst[0]; // clamp to edge boundary condition  
#pragma unroll
  for(int i=HALO+WORK;i<2*HALO+WORK;++i) buf[threadIdx.y][threadIdx.x+i*BX]=(dst_.ncols-ox>=i*BX)?dst[i*BX]:dst[dst_.ncols-ox-1]; // clamp to edge boundary condition
  // COMPUTE
  __syncthreads();
#pragma unroll
  for(int i=HALO;i<HALO+WORK;++i)
  { float sum=0.0f;
    for(int k=-radius;j<=radius;j++)
      sum+=c_kernel[radius-j]*buf[threadIdx.y][threadIdx.x+i*BX+j];
    dst[i*BX]=sum;
  }
}


template<
  typename Tdst,
  unsigned BX  , // should be 1 warp's worth
  unsigned BY  , // the number of warps to schedule for a single block
  unsigned HALO, // the number of halo elements to load per thread per side
  unsigned WORK  // each thread processes this many elements in a row
>
__global__ void 
__launch_bounds__(BX*BY,1) /*max threads,min blocks*/
  conv1_ip_cols(arg_t dst_, int radius, const nd_conv_params_t param)
{ __shared__ Tdst buf[BX][(WORK+2*HALO)*BY+1];
  Tdst* dst=(Tdst*)dst_.data;  
  const int ox=threadIdx.x+ blockIdx.x           *BX,
            oy=threadIdx.y+(blockIdx.y*WORK-HALO)*BY;
  dst+=ox+oy*dst_.rstride;
#pragma unroll
  for(int i=HALO     ;i<HALO+WORK  ;++i) buf[threadIdx.x][threadIdx.y+i*BY]=dst[i*BY*dst_.rstride];
#pragma unroll
  for(int i=0        ;i<HALO       ;++i) buf[threadIdx.x][threadIdx.y+i*BY]=(oy>=-i*BY)          ?dst[i*BY*dst_.rstride]:dst[0];  // clamp to edge boundary condition  
#pragma unroll
  for(int i=HALO+WORK;i<2*HALO+WORK;++i) buf[threadIdx.x][threadIdx.y+i*BY]=(dst_.ncols-oy>=i*BY)?dst[i*BY*dst_.rstride]:dst[(dst_.nrows-oy-1)*dst_.rstride]; // clamp to edge boundary condition
  // COMPUTE
  __syncthreads();
#pragma unroll
  for(int i<HALO;i<HALO+WORK;++i)
  { float sum=0.0f;
    for(int k=-radius;j<=radius;++j)
      sum+=c_kernel[radius-j]*buf[threadIdx.x][threadIdx.y+i*BY+j];
    dst[i*BY*dst_.rstride]=sum;
  }
}

//
// === Interface ===
//

/**
 * Assume the ndkind() of \a src_ and \a dst_ have already been checked.
 */
shared unsigned ndconv1_ip_cuda(nd_t dst_, const nd_t filter_, const unsigned idim, const nd_conv_params_t *param)
{ arg_t dst(dst_);
  unsigned radius;
  // check args
  TRY(dst.isok());
  TRY(ndtype(filter)=nd_f32); // only float kernels supported at the moment
  radius=ndnelem(filter)/2;
  TRY(2*radius+1==ndnelem(filter));      // filter has odd size
  TRY(ndnelem(filter)<MAX_FILTER_WIDTH);  

  TRY(upload_kernel(nddata(filter_),ndnelem(filter_))); /// \todo Ideally I'd only upload the kernel once and then do the seperable convolution on each dimension

  /// @cond DEFINES
  if(idim==0)
  { const unsigned BX=32,BY=4,WORK=8,HALO=1;
    dim3 blocks(dst.ncols/(WORK*BX), dst.nrows/BY);
    dim3 threads(BX,BY);
    TRY(BX*BY>=radius);                    // radius can't be too big
    TRY(dst.ncols%(WORK*BX)==0);           // width  must be aligned (8*32=256)
    TRY(dst.nrows%BY==0);                  // height must be aligned (4)
    #define CASE(T) conv1_ip_rows<T,BX,BY,HALO,WORK><<<blocks,threads>>>(dst,radius,*param); break
    TYPECASE(ndtype(dst_));
    #undef CASE 
  } else
  { 
    const unsigned BX=32,BY=8,WORK=8,HALO=1;
    dim3 blocks(dst.ncols/(WORK*BX), dst.nrows/BY);
    dim3 threads(BX,BY);
    TRY(BY*HALO>=radius);                  // radius can't be too big
    TRY(dst.ncols%BX==0);                  // width  must be aligned (32)
    TRY(dst.nrows%(BY*WORK)==0);           // height must be aligned (8*8)
    #define CASE(T) conv1_ip_cols<T,BX,BY,HALO,WORK><<<blocks,threads>>>(dst,radius,*param); break
    TYPECASE(ndtype(dst_));
    #undef CASE
  }
  /// @endcond
  CUTRY(cudaGetLastError());
  return 1;
Error:
  return 0;
}