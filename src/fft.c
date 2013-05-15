/**
 * \file
 * FFT
 *
 * GPU based FFTs are performed via CUFFT.
 *
 * TODO
 *  - It would be nice to allow for non-floating point input where possible?
 *    See http://www.mathworks.com/products/demos/fixedpt/radix2fft/fi_radix2fft_demo.html
 *    for fixed point fft on the cpu side.
 *    - I'm pretty sure CUFFT won't have fixed point support for a while.    
 */
#include "config.h"
#include "nd.h"
#include "private/kind.c"

#include "stdio.h"
#include "string.h"
#include "stdlib.h"

#include "cufft.h"
#ifdef _MSC_VER
#define alloca _alloca
#endif

// 
// Error handling
//
/// @cond DEFINES
#define ENDL       "\n"
#define LOG(...)   fprintf(stderr,__VA_ARGS__)
#define REPORT(msg1,msg2) LOG("%s(%d): %s"ENDL "\t%s"ENDL "\t%s"ENDL,__FILE__,__LINE__,__FUNCTION__,msg1,msg2)
#define TRY(e)     do{if(!(e)) {REPORT("Expression evaluated as false.",#e); goto Error; }}while(0)
#if HAVE_CUDA
  #define CUTRY(e)    do{cudaError_t ecode=(e); if(ecode!=cudaSuccess)     {REPORT(#e,cudaGetErrorString(ecode)); goto Error; }}while(0)
  #define CUFFTTRY(e) do{enum cufftResult_t ecode=(e); if(ecode!=CUFFT_SUCCESS) {REPORT(#e,cufftGetErrorString(ecode)); goto Error; }}while(0)
#else
  #define CUTRY    TRY
  #define CUFFTTRY TRY
#endif
#define TRYMSG(e,msg) do{if(!(e)) {REPORT(#e,msg); goto Error; }}while(0)
#define FAIL(msg)     do{          REPORT("FAIL",msg); goto Error; }while(0)
#define TODO          do{          REPORT("Not Implemented.","Exiting."); exit(-1); }while(0)

#define NEW(T,e,N)  ((e)=(T*)malloc(sizeof(T)*(N)))
#define ZERO(T,e,N) (memset(e,0,sizeof(T)*(N)))
/// @endcond

struct _nd_fft_plan_t
{ void (*free)(void* data);
  void *data;               ///< reference to plan
  unsigned plan_is_set;     ///< sometimes data==0 is a valid plan reference, so this flag is needed to track when free should be called.
};

nd_fft_plan_t make_plan()
{ nd_fft_plan_t plan=0;
  NEW(struct _nd_fft_plan_t,plan,1);
  ZERO(struct _nd_fft_plan_t,plan,1);
  return plan;
Error:
  return 0;
}
unsigned is_set(nd_fft_plan_t self) 
{ return self->plan_is_set;
}
nd_fft_plan_t set_plan(nd_fft_plan_t self, void *data, void (*free_fn)(void*))
{ if(is_set(self))
    self->free(self->data);
  self->data=free_fn;
  self->data=data;
  self->plan_is_set=1;
  return self;
}


// === HELPERS ===
static const char* cufftGetErrorString(enum cufftResult_t r)
{
  //extracted from CUFFT docs (CUDA v5.0)
  static const char* strings[]={
    "[ CUFFT_SUCCESS ] The CUFFT operation was successful",
    "[ CUFFT_INVALID_PLAN ] CUFFT was passed an invalid plan handle",
    "[ CUFFT_ALLOC_FAILED ] CUFFT failed to allocate GPU or CPU memory",
    "[ CUFFT_INVALID_TYPE ] No longer used",
    "[ CUFFT_INVALID_VALUE ] User specified an invalid pointer or parameter",
    "[ CUFFT_INTERNAL_ERROR ] Driver or internal CUFFT library error",
    "[ CUFFT_EXEC_FAILED ] Failed to execute an FFT on the GPU",
    "[ CUFFT_SETUP_FAILED ] The CUFFT library failed to initialize",
    "[ CUFFT_INVALID_SIZE ] User specified an invalid transform size",
    "[ CUFFT_UNALIGNED_DATA ] No longer used",
  };
  return strings[r];
}

static int prod(int* v,int n)
{ int x=1;
  while(n--) x*=v[n];
  return x;
}
static size_t prod_sz(size_t *v,int n)
{ size_t x=1;
  while(n--) x*=v[n];
  return x;
}

// === IMPLEMENTATIONS ===
// Rules:
// 0. may assume type and #dims of src and dst are the same
// 1. the "plan" container has been allocated.
//    Implementation must set the free function and set the data to free.
// 2. Implementation may copy or transpose data as required to get the 
//    complex dimension in the right place, but must undo everything so src
//    looks as if nothing happened (unless the transform is in-place).
// 3. The implementation must make sure data sizes are correct.  The caller
//    is responsible for sizing correctly, but the implementation must detect
//    errors and report.
// 4. can safely ignore the plan argument

static void ndfft_cuda_free_plan(void* data) {cufftDestroy((cufftHandle)data);}
/**
 *  \param[in,out]  dst         The destination array.
 *                              May be the same as \a src.
 *                              Type and kind must be the same as \a src.
 *  \param[in]      src         The input array.
 *                              Must be a nd_f32, or nd_f64 array.
 *  \param[in]      complex_dim The index of the dimension representing complex
 *                              values.  If complex_dim is less than 0, \a src 
 *                              is treated as a real array.
 *  \param[in]      direction   +1 for the forward transform, -1 for the inverse
 *  \param[in,out]  plan        A plan container with the cufft plan.
 *  \return dst on success, otherwise 0.
 *
 *  \todo non-zero value dimension?
 *  \todo max-dimension for rank calculation, so 1d ffts could be batched across a 3d array for example.
 *  \todo build higher rank transforms off of lower rank (may be a job for the caller)
 *  \todo validate src and dst shapes based on transfer types
 *  \todo tests
 */
static nd_t ndfft_cuda(nd_t dst, nd_t src, int direction, nd_fft_plan_t plan_)
{ cufftHandle plan=(cufftHandle)plan_->data; // will be 0 if not init'd yet
  int *shape,*ishape=0,*istrides=0,*oshape=0,*ostrides=0;
  int batch=0;
  cufftType transform_type=0;
  int svdim,  // source value dimension
      dvdim,  // dest   value dimension
      rank;   // rank before batching

  TRY( shape  =(int*)alloca( ndndim(src)   *sizeof(int))); // problem shape
  TRY(ishape  =(int*)alloca( ndndim(src)   *sizeof(int)));
  TRY(oshape  =(int*)alloca( ndndim(src)   *sizeof(int)));
  TRY(istrides=(int*)alloca((ndndim(src)+1)*sizeof(int)));
  TRY(ostrides=(int*)alloca((ndndim(src)+1)*sizeof(int)));
  ndshape_as_int(src, shape);
  ndshape_as_int(src,ishape);
  ndshape_as_int(dst,oshape);
  ndstrides_as_int(src,istrides);
  ndstrides_as_int(dst,ostrides);

  // setup the problem shape,type and rank
  { int cs,cd;
    static cufftType table[2][2][2]={0}; // keys are [is src complex][is dst complex][is float(not double)]
    if(table[1][1][1]==0) //need to init the table on the first call
    { table[0][1][0]=CUFFT_D2Z;
      table[1][0][0]=CUFFT_Z2D;
      table[1][1][0]=CUFFT_Z2Z;
      table[0][1][1]=CUFFT_R2C;
      table[1][0][1]=CUFFT_C2R;
      table[1][1][1]=CUFFT_C2C;
    }
    cs=ndshape(src)[0]==2; //complex source
    cd=ndshape(dst)[0]==2; //complex dst
    TRY(transform_type=table[cs][cd][ndtype(src)==nd_f32]);
    svdim=ndshape(src)[0]>1?1:0;
    dvdim=ndshape(dst)[0]>1?1:0;
    rank=ndndim(src)-svdim;
    rank=(rank>3)?3:rank; // max supported dim is 3, batch the rest.  FIXME: support ndims

    shape=shape+svdim;
    if(cs && !cd) //complex source, real dest
      shape[rank-1]*=2;
  }

  TRY(0<rank && rank<=3); // TODO compose low rank transforms to do higher dimensions
  // convert strides to (possibly complex) #elements
  { size_t i;
    for(i=ndndim(src);i>=svdim;--i) istrides[i]/=istrides[svdim];
    for(i=ndndim(dst);i>=dvdim;--i) ostrides[i]/=ostrides[dvdim]; 
  }

  batch=prod(ishape+svdim+rank-1,ndndim(src)-svdim-rank);
// cufft reverses the dimension order
  if(!is_set(plan_))
  { CUFFTTRY(cufftPlanMany(&plan,
      rank, // 1,2, or 3
      shape, // problem shape
      ishape,istrides[svdim],         /*stride between values*/
      istrides[svdim+rank-1],         /*stride between problems*/
      oshape,ostrides[dvdim],
      ostrides[dvdim+rank-1],
      transform_type,
      batch /* number of repititions */
      ));
    set_plan(plan_,(void*)plan,ndfft_cuda_free_plan);
  }
  CUFFTTRY(cufftExecC2C(plan,nddata(src),nddata(dst),(direction>0)?CUFFT_FORWARD:CUFFT_INVERSE));
  return dst;
Error:
  return 0;
}

extern int fft1d_ip(size_t n, nd_type_id_t tid, void *d, size_t cstride, size_t vstride, int is_inverse);

// TODO - test
// TODO - check preconditions
static nd_t ndfft_cpu(nd_t dst, nd_t src, int direction, nd_fft_plan_t plan_)
{ int cdim=0,d,D,cst;
  nd_type_id_t tid;
  uint8_t *data;
  if(nddata(dst)!=nddata(src))
    TRY(ndcopy(dst,src,0,0));
  D=ndndim(dst);
  tid=ndtype(dst);
  data=(uint8_t*)nddata(dst);
  cst=ndstrides(dst)[cdim]/ndstrides(dst)[0];
  TRY(cst==1);     // only support interleaved at the moment
  for(d=1;d<D;++d) // doing the fft on this dimension, 0 is the complex valued dimension
  { size_t o,i,
      istride=ndstrides(dst)[1],
      isize  =prod_sz(ndshape(dst),d)/2,       // product from 0 to d-1, init'd at 1. so should be 1 for d=1, divide by 2 for complex dim
      ostride=ndstrides(dst)[d+1],
      osize  =prod_sz(ndshape(dst)+d+1,D-d-1), // the rest 
      N      =ndshape(dst)[d],
      vst    =ndstrides(dst)[d]/ndstrides(dst)[0];

    for(o=0;o<osize;++o)     // iterate over outer dims
      for(i=0;i<isize;++i)   // iterate over inner dims
        TRY(fft1d_ip(N,tid,
          data+o*ostride+i*istride, //these strides are in bytes
          cst,vst,direction));      //these strides are in elements i.e. number of floats
  }
  return dst;
Error:
  return 0;
}

// === INTERFACE ===

/** FFT
 *  Possibly in place.
 *
 *  If ndshape(src)[0] is 2, then the array is treated as complex valued,
 *  otherwise it is treated as real.
 *
 *  \param[in,out]  dst         The destination array.
 *                              May be the same as \a src.
 *                              Type and kind must be the same as \a src.
 *  \param[in]      src         The input array.
 *                              Must be a nd_f32, or nd_f64 array.
 *  \param[in,out]  plan        Many fft libraries make it possible to reuse a 
 *                              "plan" setup for the transform of arrays with
 *                              the same shape.  The plan used for the last fft
 *                              can be returned and reused with this argument.
 *                              If NULL, no plan is output.  If the output plan
 *                              is not null it should be deallocated/destroyed
 *                              with ndfftFreePlan().
 *  \returns \a dst on success, or NULL otherwise.
 *  \ingroup ndops
 */
static nd_t _ndfft(nd_t dst, nd_t src, int direction, nd_fft_plan_t *plan_)
{ nd_fft_plan_t plan=plan_?*plan_:0;
  REQUIRE(src,PTR_ARITHMETIC);
  REQUIRE(dst,PTR_ARITHMETIC);
  TRY(ndtype(dst)==ndtype(src));
  TRY(ndkind(dst)==ndkind(src));
  TRY(ndtype(src)==nd_f32 || ndtype(src)==nd_f64);

  if(!plan)
    TRY(plan=make_plan());
  if(ndkind(src)==nd_gpu_cuda)
  { TRY(ndfft_cuda(dst,src,direction,plan));
  } else
  { REQUIRE(src,CAN_MEMCPY);
    TRY(ndfft_cpu(dst,src,direction,plan));
  }
  if(plan_)
    *plan_=plan;
  else
    ndfftFreePlan(plan);
  return dst;
Error:
  return NULL;
}

nd_t ndfft(nd_t dst, nd_t src, nd_fft_plan_t *plan_)
{ return _ndfft(dst,src,1,plan_); 
}

nd_t ndifft(nd_t dst, nd_t src, nd_fft_plan_t *plan_)
{ dst=_ndfft(dst,src,-1,plan_);
  return ndfmad_scalar_ip(dst,2.0f/ndnelem(src),0.0,0,0);
}

void ndfftFreePlan(nd_fft_plan_t plan)
{ if(plan && plan->free && plan->plan_is_set)
    plan->free(plan->data);
  if(plan) free(plan);
}