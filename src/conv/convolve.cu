/**
 * \file
 * nd convolution transform on the GPU with CUDA.
 *
 * \todo support different boundary conditions
 * \todo support other filter types (it's nice to have integer arithmentic sometimes)
 * \todo respect stride vs shape differences
 * \todo relax warp alignment constraints for problem shape.
 */
#include "common.h"

#define MAX_FILTER_WIDTH     32  // max size of kernel (allocated in constant memory)
                                 // actual max depends on kernel launch parameters

  /**
   * Treat the nd data as 3d around idim:
   * \a idim is the dimension to be convolved.
   * Dimensions less than \a idim are contiguous, and may be collapsed together.
   * Dimensions greater than \a idim may also be collapased together.
   */
struct arg_t
{ unsigned char *restrict data;
  int   nrows,  // It's important these are signed
        ncols,
        nplanes,
        rstride, ///< row stride. strides are in elements (not bytes)
        cstride, ///< column stride
        pstride; ///< plane stride
#undef LOG
#define LOG(...) ndLogError(a,__VA_ARGS__)
  arg_t(nd_t a,unsigned idim):
    data(0),nrows(0),ncols(0),nplanes(0),rstride(0),cstride(0)
  { 
    TRY(ndstrides(a)[0]==ndbpp(a)); // only support unit element strides
    TRY(data=(unsigned char*)nddata(a));
    cstride=1;
    if(idim==0)
    { rstride=(int)(ndstrides(a)[1]/ndstrides(a)[0]);
      pstride=1; //won't be used
      ncols=(int)ndshape(a)[0];
      nrows=(int)(ndstrides(a)[ndndim(a)]/ndstrides(a)[1]);
      nplanes=1;
    }
    else // idim>0
    { rstride=(int)(ndstrides(a)[idim]     /ndstrides(a)[0]);
      pstride=(int)(ndstrides(a)[idim+1]   /ndstrides(a)[0]);
      ncols  =(int)(ndstrides(a)[idim]     /ndstrides(a)[0]);
      nrows  =(int)(ndstrides(a)[idim+1]   /ndstrides(a)[idim]);
      nplanes=(int)(ndstrides(a)[ndndim(a)]/ndstrides(a)[idim+1]);
    }
Error:
    return; // data will be 0 if there was an error
  }
  bool isok() { return data!=0;}
};

__constant__ float c_kernel[MAX_FILTER_WIDTH];

#undef LOG
#define LOG(...) printf(__VA_ARGS__)
static unsigned upload_kernel(float *kernel,unsigned n)
{ TRY(n<MAX_FILTER_WIDTH);
  CUTRY(cudaMemcpyToSymbol(c_kernel,kernel,n*sizeof(float)));
  return 1;
Error:
  return 0;
}


template<
  typename T,
  unsigned BX  ,//=32,       // should be 1 warp's worth
  unsigned BY  ,//=4,        // the number of warps to schedule for a single block
  unsigned HALO,//=1,        // the number of halo elements to load per thread per side
  unsigned WORK //=8,        // each thread processes this many elements in a row
>
__global__ void 
__launch_bounds__(BX*BY,1) /*max threads,min blocks*/
  conv1_rows(arg_t dst_,arg_t src_,int radius, const nd_conv_params_t param)
{ 
  __shared__ T  buf[BY][(WORK+2*HALO)*BX];
             T *dst=(T*)dst_.data,
               *src=(T*)src_.data;
  const int ox=threadIdx.x+(blockIdx.x*WORK-HALO)*BX,
            oy=threadIdx.y+ blockIdx.y           *BY;

  if(oy<dst_.nrows)
  { // LOAD
    src+=ox+oy*(int)src_.rstride+(int)(blockIdx.z*src_.pstride);
    dst+=ox+oy*(int)dst_.rstride+(int)(blockIdx.z*dst_.pstride);
    #pragma unroll
    for(int i=0        ;i<HALO       ;++i) buf[threadIdx.y][threadIdx.x+i*BX]=(ox>=-i*(int)BX)?src[i*BX]:src[-ox];                 // clamp to edge boundary condition
    switch(gridDim.x-blockIdx.x)
    { 
    case 1:// last block...might hang off an unaligned edge
      #pragma unroll
      for(int i=HALO     ;i<WORK+2*HALO;++i) buf[threadIdx.y][threadIdx.x+i*BX]=(src_.ncols-ox>i*BX)?src[i*BX]:src[src_.ncols-ox-1]; // clamp to edge boundary condition
      break;
    case 2:// next to last block: bounds check end halo
      #pragma unroll
      for(int i=HALO     ;i<WORK+  HALO;++i) buf[threadIdx.y][threadIdx.x+i*BX]=src[i*BX];
      #pragma unroll
      for(int i=HALO+WORK;i<WORK+2*HALO;++i) buf[threadIdx.y][threadIdx.x+i*BX]=(src_.ncols-ox>i*BX)?src[i*BX]:src[src_.ncols-ox-1]; // clamp to edge boundary condition
      break;
    default:// not last block...everything should be in bounds
      #pragma unroll
      for(int i=HALO     ;i<WORK+2*HALO;++i) buf[threadIdx.y][threadIdx.x+i*BX]=src[i*BX];
    }
    
    // COMPUTE
    __syncthreads();
    if(blockIdx.x!=(gridDim.x-1))
    {
      #pragma unroll
      for(int i=HALO;i<HALO+WORK;++i)
      { float sum=0.0f;
        sum+=c_kernel[radius]*buf[threadIdx.y][threadIdx.x+i*BX];
        for(int j=1;j<=radius;++j)
        { sum+=c_kernel[radius-j]*buf[threadIdx.y][threadIdx.x+i*BX-j];
          sum+=c_kernel[radius+j]*buf[threadIdx.y][threadIdx.x+i*BX+j];
        }
        dst[i*BX]=sum;
      }
    } else
    { // last block 
      #pragma unroll
      for(int i=HALO;i<HALO+WORK;++i)
      { if(dst_.ncols-ox>i*BX)
        { float sum=0.0f;
          sum+=c_kernel[radius]*buf[threadIdx.y][threadIdx.x+i*BX];
          for(int j=1;j<=radius;++j)
          { sum+=c_kernel[radius-j]*buf[threadIdx.y][threadIdx.x+i*BX-j];
            sum+=c_kernel[radius+j]*buf[threadIdx.y][threadIdx.x+i*BX+j];
          }       
          dst[i*BX]=sum;
        }
      }
    }
  }
}

template<
  typename T,
  unsigned BX  , // should be 1 warp's worth
  unsigned BY  , // the number of warps to schedule for a single block
  unsigned HALO, // the number of halo elements to load per thread per side
  unsigned WORK  // each thread processes this many elements in a row
>
__global__ void 
__launch_bounds__(BX*BY,1) /*max threads,min blocks*/
  conv1_cols(arg_t dst_, arg_t src_, int radius, const nd_conv_params_t param)
{ __shared__ T buf[BX][(WORK+2*HALO)*BY+1];
             T *dst=(T*)dst_.data,
               *src=(T*)src_.data;
  const int ox=threadIdx.x+ blockIdx.x           *BX,
            oy=threadIdx.y+(blockIdx.y*WORK-HALO)*BY;
  
  if(ox<dst_.ncols)
  { src+=ox+oy*src_.rstride+(int)(blockIdx.z*src_.pstride);
    dst+=ox+oy*dst_.rstride+(int)(blockIdx.z*dst_.pstride);
  }else
  { src+=(src_.ncols-1)+oy*src_.rstride+(int)(blockIdx.z*src_.pstride); // clamp to edge boundary condition
    dst+=(dst_.ncols-1)+oy*dst_.rstride+(int)(blockIdx.z*dst_.pstride); // clamp to edge boundary condition
  }
  // LOAD
  #pragma unroll
  for(int i=0        ;i<HALO       ;++i) buf[threadIdx.x][threadIdx.y+i*BY]=(oy>=-i*(int)BY)    ?src[i*BY*src_.rstride]:src[-oy*src_.rstride];  // clamp to edge boundary condition  

  switch(gridDim.y-blockIdx.y)
  { case 1:  // last block: bounds check every access
      #pragma unroll
      for(int i=HALO     ;i<WORK+2*HALO;++i) buf[threadIdx.x][threadIdx.y+i*BY]=(src_.nrows-oy>i*BY)?src[i*BY*src_.rstride]:src[(src_.nrows-oy-1)*src_.rstride]; // clamp to edge boundary condition
      break;
    case 2:  // next to last block: bounds check end halo
      #pragma unroll
      for(int i=HALO     ;i<WORK+  HALO;++i) buf[threadIdx.x][threadIdx.y+i*BY]=src[i*BY*src_.rstride];      
      #pragma unroll
      for(int i=WORK+HALO;i<WORK+2*HALO;++i) buf[threadIdx.x][threadIdx.y+i*BY]=(src_.nrows-oy>i*BY)?src[i*BY*src_.rstride]:src[(src_.nrows-oy-1)*src_.rstride]; // clamp to edge boundary condition
      break;
    default: // no bounds checking
      #pragma unroll
      for(int i=HALO     ;i<WORK+2*HALO;++i) buf[threadIdx.x][threadIdx.y+i*BY]=src[i*BY*src_.rstride];
  }

  // COMPUTE
  __syncthreads();
  if(blockIdx.y!=(gridDim.y-1))
  {
    #pragma unroll
    for(int i=HALO;i<HALO+WORK;++i)
    { float sum=0.0f;
      sum+=c_kernel[radius]*buf[threadIdx.x][threadIdx.y+i*BY];
      for(int j=1;j<=radius;++j)
      { sum+=c_kernel[radius-j]*buf[threadIdx.x][threadIdx.y+i*BY-j];
        sum+=c_kernel[radius+j]*buf[threadIdx.x][threadIdx.y+i*BY+j];
      }
      dst[i*BY*dst_.rstride]=sum;
    }
  } else // last block
  { 
    #pragma unroll
    for(int i=HALO;i<HALO+WORK;++i)
    { if(dst_.nrows-oy>i*BY)
      { float sum=0.0f;
        sum+=c_kernel[radius]*buf[threadIdx.x][threadIdx.y+i*BY];
        for(int j=1;j<=radius;++j)
        { sum+=c_kernel[radius-j]*buf[threadIdx.x][threadIdx.y+i*BY-j];
          sum+=c_kernel[radius+j]*buf[threadIdx.x][threadIdx.y+i*BY+j];
        }
        dst[i*BY*dst_.rstride]=sum;
      }
    }
  }
}

//
// === Interface ===
//

#undef LOG
#define LOG(...) ndLogError(dst_, __VA_ARGS__)

/**
 * Assume the ndkind() of \a src_ and \a dst_ have already been checked.
 */
extern "C" unsigned ndconv1_cuda(nd_t dst_,nd_t src_,const nd_t filter_, const unsigned idim, const nd_conv_params_t *param)
{ arg_t dst(dst_,idim),
        src(src_,idim);
  unsigned radius;  
  CUTRY(cudaGetLastError());
  // check args
  TRY(param->boundary_condition==nd_boundary_replicate); // only support this boundary condition for now
  TRY(dst.isok());
  TRY(ndtype(filter_)==nd_f32); // only float kernels supported at the moment
  radius=(int)(ndnelem(filter_)/2);
  TRY(2*radius+1==ndnelem(filter_));      // filter has odd size
  TRY(ndnelem(filter_)<MAX_FILTER_WIDTH);  

  TRY(upload_kernel((float*)nddata(filter_),(unsigned)ndnelem(filter_))); /// \todo Ideally I'd only upload the kernel once and then do the seperable convolution on each dimension 
  /// @cond DEFINES
  if(idim==0)
  { //
    // ROW-WISE
    //
    // Max ncols=(WORK*BX)*MAX_BLOCK=2^8*2^16=2^24 -- max src->shape[0]
    const unsigned BX=32,BY=8,HALO=1,WORK=8;
    dim3 blocks((unsigned)ceil(src.ncols/(float)(WORK*BX)), (unsigned)ceil(src.nrows/(float)BY), src.nplanes);
    dim3 threads(BX,BY);
    int maxGridY, rem=blocks.y;
    CUTRY(cudaDeviceGetAttribute(&maxGridY,cudaDevAttrMaxGridDimY,0/*device id*/));    
    while(rem) //Process as many rows as possible per launch
    { blocks.y=min(maxGridY,rem);
    #define CASE(T) conv1_rows<T,BX,BY,HALO,WORK><<<blocks,threads,0,ndCudaStream(src_)>>>(dst,src,radius,*param); break
      {TYPECASE(ndtype(src_));} // scope just in case the compiler needs help with big switch statements.
    #undef CASE
      rem-=blocks.y;
      dst.data+=blocks.y*dst.rstride;
    }
  } else
  { //
    // COLUMN-WISE
    //
    // MAX ncols  =      BX *MAX_BLOCKS=2^5*2^16=2M -- prod(src->shape[0:i])
    // MAX nrows  =(WORK*BY)*MAX_BLOCKS=2^6*2^16=4M -- src->shape[i]
    // MAX nplanes=          MAX_BLOCKS=2^6*2^16=4M -- prod(src->shape[i:end])
    const unsigned BX=32,BY=8,WORK=8,HALO=4;
    dim3 blocks((unsigned)ceil(src.ncols/(float)BX), (unsigned)ceil(src.nrows/(float)(WORK*BY)), src.nplanes);
    dim3 threads(BX,BY);
    TRY(BY*HALO>=radius);                  // radius can't be too big
    #define CASE(T) conv1_cols<T,BX,BY,HALO,WORK><<<blocks,threads>>>(dst,src,radius,*param); break
    TYPECASE(ndtype(dst_));
    #undef CASE
  }
  /// @endcond
  CUTRY(cudaGetLastError());
  return 1;
Error:
  return 0;
}